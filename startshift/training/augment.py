from __future__ import annotations

from typing import Any

import torch

from startshift.constants import OBS_STATE
from startshift.data.state import augment_normalized_state_torch, normalize_with_dataset_stats


class RISEBatchTransform:
    def __init__(self, state_stats: dict[str, Any]):
        self.state_stats = state_stats

    def __call__(self, batch: dict[str, Any], initial_raw: torch.Tensor) -> dict[str, Any]:
        current = batch[OBS_STATE]
        initial_raw = initial_raw.to(device=current.device, dtype=current.dtype)
        initial = normalize_with_dataset_stats(initial_raw, self.state_stats)
        while initial.ndim < current.ndim:
            initial = initial.unsqueeze(1)
        if initial.shape[:-1] != current.shape[:-1]:
            initial = initial.expand(*current.shape[:-1], initial.shape[-1])
        batch[OBS_STATE] = augment_normalized_state_torch(current, initial)
        return batch


class StartShiftInferencePreprocessor:
    def __init__(self, base_preprocessor):
        self.base = base_preprocessor
        self.initial_state: torch.Tensor | None = None

    def reset(self) -> None:
        self.initial_state = None
        reset = getattr(self.base, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, transition):
        batch = self.base(transition)
        current = batch[OBS_STATE]
        if self.initial_state is None:
            self.initial_state = current.detach().clone()
        initial = self.initial_state
        while initial.ndim < current.ndim:
            initial = initial.unsqueeze(1)
        if initial.shape[:-1] != current.shape[:-1]:
            initial = initial.expand(*current.shape[:-1], initial.shape[-1])
        batch[OBS_STATE] = augment_normalized_state_torch(current, initial)
        return batch

    def __getattr__(self, name):
        return getattr(self.base, name)
