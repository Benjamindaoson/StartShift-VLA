from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from startshift.constants import ROBOTINIT_CATEGORY_ALIASES
from startshift.types import PoseRecord


def normalize_category(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    text = " ".join(text.replace("_", " ").split())
    if text in ROBOTINIT_CATEGORY_ALIASES or ("robot" in text and ("init" in text or "pose" in text)):
        return "robot"
    if "camera" in text or "view" in text:
        return "camera"
    if "language" in text:
        return "language"
    if "light" in text:
        return "light"
    if "background" in text:
        return "background"
    if "noise" in text:
        return "noise"
    if "layout" in text or "table" in text:
        return "layout"
    return text.replace(" ", "_")


def find_classification_file(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    env = os.environ.get("STARTSHIFT_LIBERO_PLUS_CLASSIFICATION")
    if env:
        return find_classification_file(env)

    candidates: list[Path] = []
    for root in (Path("third_party/LIBERO-plus"), Path.home() / ".libero"):
        if root.exists():
            candidates.extend(root.rglob("task_classification.json"))

    try:
        import libero

        pkg = Path(libero.__file__).resolve().parent
        candidates.extend(pkg.parent.rglob("task_classification.json"))
    except (ImportError, TypeError):
        pass

    unique = sorted({p.resolve() for p in candidates if p.is_file()})
    if not unique:
        raise FileNotFoundError(
            "Could not locate task_classification.json. Set "
            "STARTSHIFT_LIBERO_PLUS_CLASSIFICATION or pass --classification."
        )
    if len(unique) > 1:
        preferred = [p for p in unique if "LIBERO-plus" in str(p)]
        if len(preferred) == 1:
            return preferred[0]
    return unique[0]


def _looks_like_record(value: dict[str, Any]) -> bool:
    lowered = {str(k).lower() for k in value}
    has_id = bool(lowered & {"task_id", "taskid", "id", "index"})
    has_category = bool(lowered & {"category", "perturbation", "type", "dimension"})
    return has_id and has_category


def _get_first(value: dict[str, Any], keys: Iterable[str]) -> Any:
    by_lower = {str(k).lower(): v for k, v in value.items()}
    for key in keys:
        if key in by_lower:
            return by_lower[key]
    return None


def _record_from_mapping(value: dict[str, Any], suite_hint: str | None = None) -> PoseRecord:
    task_id = _get_first(value, ("task_id", "taskid", "id", "index"))
    category = _get_first(value, ("category", "perturbation", "type", "dimension"))
    difficulty = _get_first(value, ("difficulty", "level", "severity"))
    suite = _get_first(value, ("suite", "benchmark", "task_suite")) or suite_hint or "unknown"
    base_task = _get_first(value, ("base_task", "base_task_id", "original_task", "task_name"))
    pose_id = _get_first(value, ("pose_id", "variant_id", "reset_id"))
    if task_id is None or category is None:
        raise ValueError(f"Not a task classification record: {value}")
    return PoseRecord(
        suite=str(suite),
        task_id=int(task_id),
        category=normalize_category(category),
        difficulty=difficulty,
        base_task=base_task,
        pose_id=None if pose_id is None else str(pose_id),
        metadata=value,
    )


def _walk_schema(node: Any, path: tuple[str, ...] = ()) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    if isinstance(node, list):
        for item in node:
            records.extend(_walk_schema(item, path))
        return records

    if not isinstance(node, dict):
        return records

    if _looks_like_record(node):
        try:
            records.append(_record_from_mapping(node, path[0] if path else None))
        except (TypeError, ValueError):
            pass
        return records

    # Common schema: {suite: {category: {L1: [ids]}}}
    for key, value in node.items():
        key_text = str(key)
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(x, (int, str)) and str(x).lstrip("-").isdigit() for x in value
        ):
            category = None
            difficulty = None
            for part in reversed(path + (key_text,)):
                norm = normalize_category(part)
                if norm in {"robot", "camera", "language", "light", "background", "noise", "layout"}:
                    category = norm
                    break
            for part in reversed(path + (key_text,)):
                p = str(part).lower()
                if p.startswith("l") and p[1:].isdigit():
                    difficulty = p.upper()
                    break
                if p.isdigit():
                    difficulty = int(p)
                    break
            if category is not None:
                suite = path[0] if path else "unknown"
                for task_id in value:
                    records.append(
                        PoseRecord(
                            suite=str(suite),
                            task_id=int(task_id),
                            category=category,
                            difficulty=difficulty,
                        )
                    )
                continue
        records.extend(_walk_schema(value, path + (key_text,)))
    return records


def load_pose_records(
    path: str | Path | None = None,
    *,
    category: str | None = "robot",
) -> list[PoseRecord]:
    source = find_classification_file(path)
    data = json.loads(source.read_text())
    records = _walk_schema(data)
    if not records:
        raise ValueError(
            f"No task records could be parsed from {source}. "
            "Pass a normalized manifest generated by StartShift if upstream changed schema."
        )
    dedup: dict[tuple[str, int, str, str], PoseRecord] = {}
    for record in records:
        key = (record.suite, record.task_id, record.category, str(record.difficulty))
        dedup[key] = record
    values = sorted(dedup.values(), key=lambda r: (r.suite, r.task_id, str(r.difficulty)))
    if category is not None:
        wanted = normalize_category(category)
        values = [r for r in values if r.category == wanted]
    return values
