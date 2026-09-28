from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from startshift.types import EvalRecord


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def lower_tail_cvar(values: Iterable[float], alpha: float = 0.2) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    k = max(1, int(math.ceil(alpha * arr.size)))
    return float(np.sort(arr)[:k].mean())


def _group_success(records: list[EvalRecord], key_fn) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        groups[str(key_fn(record))].append(int(record.success))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def summarize_records(records: Iterable[EvalRecord], *, cvar_alpha: float = 0.2) -> dict:
    rows = list(records)
    n = len(rows)
    successes = sum(int(r.success) for r in rows)
    mean_sr = successes / n if n else float("nan")
    ci_low, ci_high = wilson_interval(successes, n)

    pose_sr = _group_success(rows, lambda r: r.pose_id or f"{r.suite}:{r.task_id}")
    difficulty_sr = _group_success(rows, lambda r: r.difficulty)
    pose_values = list(pose_sr.values())

    return {
        "n_episodes": n,
        "successes": successes,
        "success_rate": mean_sr,
        "wilson95": [ci_low, ci_high],
        "pose_success": pose_sr,
        "difficulty_success": difficulty_sr,
        "worst_pose_success": min(pose_values) if pose_values else float("nan"),
        "cvar20_pose_success": lower_tail_cvar(pose_values, cvar_alpha),
        "pose_success_std": float(np.std(pose_values)) if pose_values else float("nan"),
    }


def robustness_retention(robotinit_success: float, id_success: float) -> float:
    if id_success <= 0:
        return float("nan")
    return robotinit_success / id_success


def adaptation_efficiency(
    gain: float,
    *,
    adaptation_samples: int,
    trainable_parameters: int,
    gpu_hours: float | None = None,
) -> dict[str, float]:
    result = {
        "gain_per_100_samples": gain / max(adaptation_samples, 1) * 100.0,
        "gain_per_million_trainable_params": gain / max(trainable_parameters / 1_000_000.0, 1e-9),
    }
    if gpu_hours is not None:
        result["gain_per_gpu_hour"] = gain / max(gpu_hours, 1e-9)
    return result
