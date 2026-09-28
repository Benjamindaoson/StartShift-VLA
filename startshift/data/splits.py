from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from startshift.constants import DEFAULT_ADAPT_BUDGETS, DEFAULT_SEED
from startshift.types import PoseRecord
from startshift.utils.io import read_json, write_json


def _round_robin_stratified(records: list[PoseRecord], seed: int) -> list[PoseRecord]:
    rng = random.Random(seed)
    groups: dict[str, list[PoseRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)
    for group in groups.values():
        rng.shuffle(group)

    ordered: list[PoseRecord] = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def _difficulty_rank(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else 0.0


def validate_disjoint_splits(splits: dict[str, list[PoseRecord]]) -> None:
    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        if split_name.startswith("adapt_"):
            continue
        for record in records:
            group = record.group_id
            if group in seen:
                raise ValueError(f"Pose group {group!r} leaked between {seen[group]!r} and {split_name!r}.")
            seen[group] = split_name


def build_pose_splits(
    records: Iterable[PoseRecord],
    *,
    seed: int = DEFAULT_SEED,
    audit_fraction: float = 0.2,
    dev_fraction: float = 0.2,
    adapt_pool_fraction: float = 0.3,
    adapt_budgets: tuple[int, ...] = DEFAULT_ADAPT_BUDGETS,
) -> dict[str, list[PoseRecord]]:
    """Create deterministic RobotInit splits with nested random and targeted budgets."""
    values = list(records)
    if len(values) < 5:
        raise ValueError("Need at least five RobotInit records to create meaningful splits.")

    ordered = _round_robin_stratified(values, seed)
    n = len(ordered)
    n_audit = max(1, round(n * audit_fraction))
    n_dev = max(1, round(n * dev_fraction))
    n_adapt = max(1, round(n * adapt_pool_fraction))

    if n_audit + n_dev + n_adapt >= n:
        overflow = n_audit + n_dev + n_adapt - (n - 1)
        n_adapt = max(1, n_adapt - overflow)

    audit = ordered[:n_audit]
    adapt_pool = ordered[n_audit : n_audit + n_adapt]
    dev = ordered[n_audit + n_adapt : n_audit + n_adapt + n_dev]
    heldout = ordered[n_audit + n_adapt + n_dev :]
    if not heldout:
        raise ValueError("Held-out split is empty. Reduce audit/dev/adapt fractions.")

    result: dict[str, list[PoseRecord]] = {
        "audit": audit,
        "adapt_pool": adapt_pool,
        "dev": dev,
        "heldout_test": heldout,
    }
    targeted = sorted(
        adapt_pool,
        key=lambda r: (_difficulty_rank(r.difficulty), r.suite, r.task_id),
        reverse=True,
    )
    for budget in sorted(set(int(x) for x in adapt_budgets)):
        k = min(budget, len(adapt_pool))
        result[f"adapt_{budget}"] = adapt_pool[:k]
        result[f"adapt_random_{budget}"] = adapt_pool[:k]
        result[f"adapt_difficulty_{budget}"] = targeted[:k]

    validate_disjoint_splits(
        {k: v for k, v in result.items() if k in {"audit", "adapt_pool", "dev", "heldout_test"}}
    )
    return result


def save_pose_splits(splits: dict[str, list[PoseRecord]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for name, records in splits.items():
        write_json([record.to_dict() for record in records], out / f"{name}.json")
        summary[name] = len(records)
    write_json(summary, out / "summary.json")


def load_pose_split(path: str | Path) -> list[PoseRecord]:
    return [PoseRecord.from_dict(row) for row in read_json(path)]



def build_failure_targeted_split(
    pool: Iterable[PoseRecord],
    eval_records: Iterable[dict],
    *,
    budget: int,
) -> list[PoseRecord]:
    """Select the lowest-performing RobotInit groups using *pool-only* base-policy evaluation.

    This implements a genuine targeted-augmentation baseline without test leakage.
    Every scored group must belong to the supplied adaptation pool. The held-out
    split must never be passed to this function.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    pool_records = list(pool)
    by_group = {record.group_id: record for record in pool_records}
    if not by_group:
        raise ValueError("adaptation pool is empty")

    scores: dict[str, list[int]] = defaultdict(list)
    for row in eval_records:
        group = str(row.get("pose_id") or f"{row.get('suite')}:{row.get('task_id')}")
        if group not in by_group:
            raise ValueError(
                f"Evaluation record {group!r} is outside the adaptation pool; "
                "refusing targeted selection to prevent test leakage."
            )
        scores[group].append(int(bool(row.get("success"))))

    missing = set(by_group) - set(scores)
    if missing:
        raise ValueError(
            "Targeted selection requires a base-policy score for every pool group; "
            f"missing {len(missing)} groups."
        )

    def key(record: PoseRecord):
        values = scores[record.group_id]
        sr = sum(values) / len(values)
        # Lower empirical success is harder. Break ties by official difficulty,
        # then stable identifiers for deterministic selection.
        return (sr, -_difficulty_rank(record.difficulty), record.suite, record.task_id)

    ordered = sorted(pool_records, key=key)
    return ordered[: min(budget, len(ordered))]


def save_failure_targeted_splits(
    pool_path: str | Path,
    eval_records_path: str | Path,
    out_dir: str | Path,
    *,
    budgets: Iterable[int] = DEFAULT_ADAPT_BUDGETS,
) -> dict[str, int]:
    from startshift.utils.io import read_jsonl

    pool = load_pose_split(pool_path)
    eval_rows = read_jsonl(eval_records_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for budget in sorted(set(int(x) for x in budgets)):
        records = build_failure_targeted_split(pool, eval_rows, budget=budget)
        name = f"adapt_targeted_{budget}"
        write_json([record.to_dict() for record in records], out / f"{name}.json")
        summary[name] = len(records)
    write_json(summary, out / "targeted_summary.json")
    return summary
