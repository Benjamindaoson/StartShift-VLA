import pytest

from startshift.data.splits import (
    build_failure_targeted_split,
    build_pose_splits,
    validate_disjoint_splits,
)
from startshift.types import PoseRecord


def _records(n=80):
    return [
        PoseRecord(
            suite="libero_spatial",
            task_id=i,
            category="robot",
            difficulty=f"L{1 + (i % 5)}",
            base_task=i % 10,
            pose_id=f"pose-{i}",
        )
        for i in range(n)
    ]


def test_pose_splits_are_reproducible_and_disjoint():
    a = build_pose_splits(_records(), seed=7)
    b = build_pose_splits(_records(), seed=7)
    assert [x.group_id for x in a["heldout_test"]] == [x.group_id for x in b["heldout_test"]]
    core = ["audit", "adapt_pool", "dev", "heldout_test"]
    sets = {name: {r.group_id for r in a[name]} for name in core}
    for i, left in enumerate(core):
        for right in core[i + 1 :]:
            assert sets[left].isdisjoint(sets[right])


def test_core_splits_preserve_uneven_strata():
    records = [
        PoseRecord("rare", i, "robot", "L1", pose_id=f"rare-{i}") for i in range(5)
    ] + [
        PoseRecord("common", i, "robot", "L2", pose_id=f"common-{i}") for i in range(15)
    ]
    splits = build_pose_splits(records, seed=7)
    expected = {records[0].stratum, records[-1].stratum}
    for name in ("audit", "adapt_pool", "dev", "heldout_test"):
        assert {record.stratum for record in splits[name]} == expected


def test_disjoint_validator_checks_adaptation_pool():
    leaked = PoseRecord("s", 1, "robot", "L1", pose_id="shared")
    with pytest.raises(ValueError, match="leaked"):
        validate_disjoint_splits(
            {"audit": [leaked], "adapt_pool": [leaked], "dev": [], "heldout_test": []}
        )


def test_adaptation_budgets_are_nested_and_difficulty_split_is_hard_first():
    splits = build_pose_splits(_records(), seed=42, adapt_budgets=(5, 10))
    assert {r.group_id for r in splits["adapt_random_5"]}.issubset(
        {r.group_id for r in splits["adapt_random_10"]}
    )
    ranks = [int(str(r.difficulty).lstrip("L")) for r in splits["adapt_difficulty_10"]]
    assert ranks == sorted(ranks, reverse=True)


def test_failure_targeted_split_uses_empirical_failure_rate():
    pool = [
        PoseRecord("s", 1, "robot", "L1", pose_id="a"),
        PoseRecord("s", 2, "robot", "L5", pose_id="b"),
        PoseRecord("s", 3, "robot", "L3", pose_id="c"),
    ]
    rows = []
    # a: 100%, b: 50%, c: 0% => c should be selected first despite b being L5.
    for pose, task, values in [("a", 1, [1, 1]), ("b", 2, [1, 0]), ("c", 3, [0, 0])]:
        for success in values:
            rows.append({"pose_id": pose, "suite": "s", "task_id": task, "success": success})
    selected = build_failure_targeted_split(pool, rows, budget=2)
    assert [r.group_id for r in selected] == ["c", "b"]


def test_failure_targeted_split_rejects_test_leakage():
    pool = [PoseRecord("s", 1, "robot", "L1", pose_id="a")]
    rows = [{"pose_id": "heldout", "suite": "s", "task_id": 99, "success": False}]
    try:
        build_failure_targeted_split(pool, rows, budget=1)
    except ValueError as exc:
        assert "outside the adaptation pool" in str(exc)
    else:
        raise AssertionError("Expected leakage guard to reject an out-of-pool record")
