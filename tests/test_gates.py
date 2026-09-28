import pytest

from startshift.gates import GateThresholds, evaluate_method_gate, evaluate_phenomenon_gate


def _summary(sr, tail=None):
    value = {"success_rate": sr}
    if tail is not None:
        value["cvar20_pose_success"] = tail
    return value


def test_phenomenon_gate():
    result = evaluate_phenomenon_gate(
        id_summary=_summary(0.80),
        robotinit_summary=_summary(0.55),
        thresholds=GateThresholds(phenomenon_gap_pp=15),
    )
    assert result.passed
    assert result.values["distribution_shift_gap_pp"] == pytest.approx(25.0)


def test_method_gate_requires_gain_id_preservation_and_tail():
    result = evaluate_method_gate(
        baseline_heldout=_summary(0.55, 0.20),
        method_heldout=_summary(0.61, 0.27),
        baseline_id=_summary(0.80),
        method_id=_summary(0.79),
        thresholds=GateThresholds(min_heldout_gain_pp=3, max_id_drop_pp=2, min_tail_gain_pp=3),
    )
    assert result.passed


def test_method_gate_rejects_id_tradeoff():
    result = evaluate_method_gate(
        baseline_heldout=_summary(0.55, 0.20),
        method_heldout=_summary(0.65, 0.30),
        baseline_id=_summary(0.80),
        method_id=_summary(0.75),
    )
    assert not result.passed
    assert not result.checks["id_preservation"]
