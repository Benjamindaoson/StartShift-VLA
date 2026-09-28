from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from startshift.constants import FRAME_INDEX, OBS_STATE, TASK_INDEX


def extract_task_initial_states(dataset) -> pd.DataFrame:
    rows: list[dict] = []
    table = dataset.hf_dataset.select_columns([TASK_INDEX, FRAME_INDEX, OBS_STATE])
    for row in table:
        if int(row[FRAME_INDEX]) != 0:
            continue
        state = np.asarray(row[OBS_STATE], dtype=np.float64)
        item = {"task_id": int(row[TASK_INDEX])}
        for i, value in enumerate(state):
            item[f"s{i}"] = float(value)
        rows.append(item)
    if not rows:
        raise ValueError("No frame_index=0 states found.")
    return pd.DataFrame(rows)


def pose_distance_table(initial_states: pd.DataFrame, reference_task: int | None = None) -> pd.DataFrame:
    state_cols = [c for c in initial_states.columns if c.startswith("s")]
    means = initial_states.groupby("task_id", as_index=False)[state_cols].mean()
    if reference_task is None:
        reference = means[state_cols].mean(axis=0).to_numpy()
    else:
        selected = means.loc[means["task_id"] == int(reference_task), state_cols]
        if selected.empty:
            raise ValueError(f"reference_task={reference_task} is not present")
        reference = selected.iloc[0].to_numpy()
    matrix = means[state_cols].to_numpy()
    means["distance_to_reference"] = np.linalg.norm(matrix - reference[None, :], axis=1)
    return means


def write_pose_distance_csv(dataset, output_path: str | Path, reference_task: int | None = None) -> Path:
    table = pose_distance_table(extract_task_initial_states(dataset), reference_task=reference_task)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
