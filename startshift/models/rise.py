from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(slots=True)
class RISEProjectorConfig:
    state_dim: int = 8
    mode: str = "linear"
    bottleneck_dim: int = 128
    residual_scale: float = 1.0
    freeze_base: bool = True


class RISEStateProjector(nn.Module):
    """Drop-in replacement for SmolVLA state projection.

    Input layout before SmolVLA padding is:
    [current_state, initial_state, current_state - initial_state].

    The pretrained base branch explicitly consumes only current_state. The
    zero-initialized context branch learns reset-state information, so a newly
    patched policy starts behaviorally equivalent to the pretrained policy.
    """

    def __init__(self, base: nn.Linear, config: RISEProjectorConfig):
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.mode not in {"linear", "adapter"}:
            raise ValueError(f"Unknown RISE projector mode: {config.mode}")
        if base.in_features < 3 * config.state_dim:
            raise ValueError(
                f"Base width {base.in_features} is too small for 3*state_dim={3 * config.state_dim}."
            )
        self.base = base
        self.config = config
        context_dim = 2 * config.state_dim
        hidden = base.out_features

        if config.mode == "linear":
            self.context = nn.Linear(context_dim, hidden, bias=False)
            nn.init.zeros_(self.context.weight)
        else:
            bottleneck = max(1, int(config.bottleneck_dim))
            self.context = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, bottleneck),
                nn.SiLU(),
                nn.Linear(bottleneck, hidden),
            )
            final = self.context[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        if config.freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] < 3 * self.state_dim:
            raise ValueError(
                f"RISE needs at least {3*self.state_dim} state features, got {state.shape[-1]}."
            )
        base_input = torch.zeros_like(state)
        base_input[..., : self.state_dim] = state[..., : self.state_dim]
        base_out = self.base(base_input)
        context = state[..., self.state_dim : 3 * self.state_dim]
        residual = self.context(context)
        return base_out + float(self.config.residual_scale) * residual

    def manifest(self) -> dict:
        return asdict(self.config)


def patch_smolvla_state_projector(
    policy: nn.Module,
    *,
    mode: str,
    state_dim: int = 8,
    bottleneck_dim: int = 128,
    residual_scale: float = 1.0,
    freeze_base: bool = True,
) -> RISEStateProjector:
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "state_proj"):
        raise TypeError("Expected a SmolVLA-like policy with policy.model.state_proj.")
    if isinstance(model.state_proj, RISEStateProjector):
        return model.state_proj
    if not isinstance(model.state_proj, nn.Linear):
        raise TypeError(f"Expected nn.Linear state_proj, got {type(model.state_proj)!r}.")
    projector = RISEStateProjector(
        model.state_proj,
        RISEProjectorConfig(
            state_dim=state_dim,
            mode=mode,
            bottleneck_dim=bottleneck_dim,
            residual_scale=residual_scale,
            freeze_base=freeze_base,
        ),
    )
    model.state_proj = projector
    return projector


def freeze_all_except_rise(policy: nn.Module) -> list[nn.Parameter]:
    for parameter in policy.parameters():
        parameter.requires_grad = False
    projector = policy.model.state_proj
    if not isinstance(projector, RISEStateProjector):
        raise TypeError("Policy is not patched with RISEStateProjector.")
    for parameter in projector.context.parameters():
        parameter.requires_grad = True
    return [p for p in policy.parameters() if p.requires_grad]


def count_parameters(module: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }
