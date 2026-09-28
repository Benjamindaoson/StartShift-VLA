from __future__ import annotations

import csv
from enum import StrEnum
from pathlib import Path

from startshift.types import EvalRecord
from startshift.utils.io import read_jsonl, write_jsonl


class FailureType(StrEnum):
    TARGET_PERCEPTION = "TARGET_PERCEPTION"
    GEOMETRY_ACTION_GROUNDING = "GEOMETRY_ACTION_GROUNDING"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    CONTROL = "CONTROL"
    RECOVERY = "RECOVERY"
    TIMEOUT = "TIMEOUT"
    OTHER = "OTHER"
    UNLABELED = "UNLABELED"


def write_annotation_template(records_path: str | Path, output_csv: str | Path) -> Path:
    records = [EvalRecord.from_dict(x) for x in read_jsonl(records_path)]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row", "method", "suite", "task_id", "episode", "pose_id", "failure_type", "notes"],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            if record.success:
                continue
            writer.writerow(
                {
                    "row": index,
                    "method": record.method,
                    "suite": record.suite,
                    "task_id": record.task_id,
                    "episode": record.episode,
                    "pose_id": record.pose_id or "",
                    "failure_type": record.failure_type or FailureType.UNLABELED,
                    "notes": "",
                }
            )
    return path


def apply_annotations(
    records_path: str | Path,
    annotations_csv: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    raw = read_jsonl(records_path)
    with Path(annotations_csv).open() as handle:
        annotations = {int(row["row"]): row for row in csv.DictReader(handle)}
    for index, row in enumerate(raw):
        if index not in annotations:
            continue
        label = annotations[index]["failure_type"].strip()
        if label:
            row["failure_type"] = FailureType(label).value
        notes = annotations[index].get("notes", "").strip()
        if notes:
            row.setdefault("metadata", {})
            row["metadata"]["failure_notes"] = notes
    target = Path(output_path or records_path)
    write_jsonl(raw, target)
    return target
