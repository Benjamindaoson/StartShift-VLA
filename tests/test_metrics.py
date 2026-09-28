import pytest

from startshift.evaluation.metrics import lower_tail_cvar, summarize_records, wilson_interval
from startshift.types import EvalRecord


def test_lower_tail_cvar():
    assert lower_tail_cvar([0.1,0.2,0.8,0.9,1.0], 0.4) == pytest.approx(0.15)


def test_summary_reports_mean_and_tail():
    rows=[]
    for pose, success_count in [("a",4),("b",2),("c",0)]:
        for episode in range(4):
            rows.append(EvalRecord(method="base",suite="s",task_id=0,episode=episode,pose_id=pose,success=episode < success_count))
    result=summarize_records(rows)
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["worst_pose_success"] == 0.0
    assert result["cvar20_pose_success"] == 0.0


def test_wilson_interval_is_bounded():
    low,high=wilson_interval(7,10)
    assert 0 <= low <= 0.7 <= high <= 1
