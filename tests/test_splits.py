from startshift.data.splits import build_pose_splits
from startshift.types import PoseRecord


def _records(n=80):
    return [
        PoseRecord(
            suite="libero_spatial", task_id=i, category="robot",
            difficulty=f"L{1 + (i % 5)}", base_task=i % 10, pose_id=f"pose-{i}"
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
        for right in core[i + 1:]:
            assert sets[left].isdisjoint(sets[right])


def test_adaptation_budgets_are_nested_and_targeted_is_hard_first():
    splits = build_pose_splits(_records(), seed=42, adapt_budgets=(5, 10))
    assert {r.group_id for r in splits["adapt_random_5"]}.issubset(
        {r.group_id for r in splits["adapt_random_10"]}
    )
    ranks = [int(str(r.difficulty).lstrip("L")) for r in splits["adapt_targeted_10"]]
    assert ranks == sorted(ranks, reverse=True)
