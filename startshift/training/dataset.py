from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from startshift.constants import (
    EPISODE_INDEX,
    FRAME_INDEX,
    OBS_STATE,
    STARTSHIFT_GROUP_INDEX,
    STARTSHIFT_INITIAL_STATE,
    TASK_INDEX,
)
from startshift.types import PoseRecord


def episodes_for_task_ids(meta, task_ids: Iterable[int], *, limit_per_task: int | None = None) -> list[int]:
    wanted = {int(x) for x in task_ids}
    by_task: dict[int, list[int]] = defaultdict(list)
    for row in meta.episodes:
        task_index = int(row["task_index"])
        if task_index in wanted:
            by_task[task_index].append(int(row["episode_index"]))
    missing = wanted - set(by_task)
    if missing:
        raise ValueError(f"No dataset episodes found for task ids: {sorted(missing)[:20]}")
    selected: list[int] = []
    for task_id in sorted(wanted):
        values = sorted(by_task[task_id])
        selected.extend(values if limit_per_task is None else values[:limit_per_task])
    return selected


def difficulty_group_map(records: Iterable[PoseRecord]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for record in records:
        text = str(record.difficulty or "1").upper().lstrip("L")
        try:
            value = int(float(text))
        except ValueError:
            value = 1
        mapping[int(record.task_id)] = max(0, min(4, value - 1))
    return mapping


def build_initial_state_cache(dataset) -> dict[int, torch.Tensor]:
    table = dataset.hf_dataset.select_columns([EPISODE_INDEX, FRAME_INDEX, OBS_STATE])
    cache: dict[int, torch.Tensor] = {}
    for row in table:
        if int(row[FRAME_INDEX]) != 0:
            continue
        ep = int(row[EPISODE_INDEX])
        cache[ep] = torch.as_tensor(np.asarray(row[OBS_STATE], dtype=np.float32))
    selected = set(int(x) for x in (dataset.episodes or cache.keys()))
    missing = selected - set(cache)
    if missing:
        raise ValueError(f"Could not find frame_index=0 for episodes: {sorted(missing)[:20]}")
    return cache


class InitialStateDataset(Dataset):
    def __init__(self, dataset, group_by_task: dict[int, int] | None = None):
        self.dataset = dataset
        self.meta = dataset.meta
        self.initial_states = build_initial_state_cache(dataset)
        self.group_by_task = group_by_task or {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        episode = int(item[EPISODE_INDEX])
        task = int(item[TASK_INDEX])
        item[STARTSHIFT_INITIAL_STATE] = self.initial_states[episode].clone()
        item[STARTSHIFT_GROUP_INDEX] = torch.tensor(self.group_by_task.get(task, 0), dtype=torch.long)
        return item


def load_subset_for_policy(
    *,
    repo_id: str,
    policy_config,
    task_ids: Iterable[int],
    revision: str | None = None,
    limit_per_task: int | None = None,
    return_uint8: bool = True,
):
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(repo_id, revision=revision)
    episodes = episodes_for_task_ids(meta, task_ids, limit_per_task=limit_per_task)
    deltas = resolve_delta_timestamps(policy_config, meta)
    dataset = LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps=deltas,
        revision=revision,
        return_uint8=return_uint8,
    )
    return dataset, episodes
