from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_ADAPT_BUDGETS, DEFAULT_SEED


@dataclass(slots=True)
class DataConfig:
    dataset_repo: str = "lerobot/libero_plus"
    classification_path: str | None = None
    split_dir: str = "splits"
    category: str = "robot"
    seed: int = DEFAULT_SEED
    adapt_budgets: tuple[int, ...] = DEFAULT_ADAPT_BUDGETS


@dataclass(slots=True)
class PolicyConfig:
    base_policy: str = "lerobot/smolvla_libero"
    method: str = "base"
    state_dim: int = 8
    bottleneck_dim: int = 128
    adapter_scale: float = 1.0
    freeze_base_state_proj: bool = True


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "outputs/train/rise"
    steps: int = 10_000
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    log_every: int = 20
    save_every: int = 2_000
    seed: int = DEFAULT_SEED
    device: str = "cuda"
    bf16: bool = True


@dataclass(slots=True)
class EvalConfig:
    output_dir: str = "outputs/eval"
    episodes_per_task: int = 10
    batch_size: int = 1
    seed: int = DEFAULT_SEED
    hard_reset: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "startshift"
    data: DataConfig = field(default_factory=DataConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        return cls(
            name=value.get("name", "startshift"),
            data=DataConfig(**value.get("data", {})),
            policy=PolicyConfig(**value.get("policy", {})),
            train=TrainConfig(**value.get("train", {})),
            eval=EvalConfig(**value.get("eval", {})),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    value = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig.from_dict(value)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    from dataclasses import asdict

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
