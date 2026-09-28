from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class MeanStdStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> MeanStdStats:
        mean = np.asarray(value["mean"], dtype=np.float32)
        std = np.asarray(value["std"], dtype=np.float32)
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.std + self.mean


def augment_normalized_state(
    current_normalized: np.ndarray,
    initial_normalized: np.ndarray,
) -> np.ndarray:
    """RISE-E representation: [current, initial, current-initial].

    SmolVLA uses mean/std state normalization.  Under the same affine
    normalization, current_normalized - initial_normalized is exactly the raw
    state difference divided by the training standard deviation.  Keeping all
    three terms preserves absolute geometry while exposing the reset-state
    relation explicitly.
    """

    current = np.asarray(current_normalized, dtype=np.float32)
    initial = np.asarray(initial_normalized, dtype=np.float32)
    if current.shape != initial.shape:
        raise ValueError(f"State shape mismatch: current={current.shape}, initial={initial.shape}")
    return np.concatenate([current, initial, current - initial], axis=-1)


def augment_normalized_state_torch(current, initial):
    import torch

    if not isinstance(current, torch.Tensor) or not isinstance(initial, torch.Tensor):
        raise TypeError("current and initial must be torch tensors")
    if current.shape != initial.shape:
        raise ValueError(f"State shape mismatch: current={current.shape}, initial={initial.shape}")
    return torch.cat([current, initial, current - initial], dim=-1)


def normalize_with_dataset_stats(value, stats: dict[str, Any]):
    """Normalize a state using LeRobot's mean/std metadata without importing LeRobot."""

    import torch

    mean = torch.as_tensor(stats["mean"], dtype=value.dtype, device=value.device)
    std = torch.as_tensor(stats["std"], dtype=value.dtype, device=value.device)
    std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
    while mean.ndim < value.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (value - mean) / std
