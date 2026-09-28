from __future__ import annotations

import torch


class GroupDRO:
    def __init__(self, num_groups: int = 5, eta: float = 0.05, device: str | torch.device = "cpu"):
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.num_groups = num_groups
        self.eta = float(eta)
        self.weights = torch.ones(num_groups, device=device, dtype=torch.float32) / num_groups

    @torch.no_grad()
    def _update(self, group_losses: torch.Tensor, present: torch.Tensor) -> None:
        safe = torch.where(present, group_losses, torch.zeros_like(group_losses))
        self.weights = self.weights * torch.exp(self.eta * safe)
        self.weights = self.weights / self.weights.sum().clamp_min(1e-12)

    def __call__(self, per_sample_loss: torch.Tensor, groups: torch.Tensor):
        if per_sample_loss.ndim != 1:
            raise ValueError("per_sample_loss must have shape [batch]")
        groups = groups.to(per_sample_loss.device).long()
        if groups.shape != per_sample_loss.shape:
            raise ValueError("groups and per_sample_loss must have identical shape")
        group_losses = torch.zeros(self.num_groups, device=per_sample_loss.device)
        present = torch.zeros(self.num_groups, dtype=torch.bool, device=per_sample_loss.device)
        for group in range(self.num_groups):
            mask = groups == group
            if mask.any():
                group_losses[group] = per_sample_loss[mask].mean()
                present[group] = True
        self.weights = self.weights.to(per_sample_loss.device)
        self._update(group_losses.detach(), present)
        active = self.weights * present.float()
        active = active / active.sum().clamp_min(1e-12)
        loss = (active * group_losses).sum()
        return loss, {
            "group_weights": self.weights.detach().cpu().tolist(),
            "group_losses": group_losses.detach().cpu().tolist(),
            "present_groups": present.detach().cpu().tolist(),
        }

    def state_dict(self) -> dict:
        return {"weights": self.weights.detach().cpu(), "eta": self.eta, "num_groups": self.num_groups}

    def load_state_dict(self, value: dict) -> None:
        self.eta = float(value["eta"])
        self.num_groups = int(value["num_groups"])
        self.weights = value["weights"].clone()
