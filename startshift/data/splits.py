from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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


def validate_disjoint_splits(splits: dict[str, list[PoseRecord]]) -> None:
    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        # Nested adaptation budgets intentionally overlap each other.
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
    """Create deterministic, group-disjoint RobotInit splits.

    Adaptation budgets are nested subsets of the adaptation pool.  Dev and test
    contain pose groups never used for adaptation.
    """

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
    for budget in sorted(set(int(x) for x in adapt_budgets)):
        result[f"adapt_{budget}"] = adapt_pool[: min(budget, len(adapt_pool))]

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
