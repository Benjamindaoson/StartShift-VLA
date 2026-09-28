from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from startshift.utils.io import read_json, write_json


@dataclass(slots=True)
class GateThresholds:
    phenomenon_gap_pp: float = 15.0
    min_heldout_gain_pp: float = 3.0
    max_id_drop_pp: float = 2.0
    min_tail_gain_pp: float = 3.0


@dataclass(slots=True)
class GateResult:
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float | None]
    thresholds: GateThresholds
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["thresholds"] = asdict(self.thresholds)
        return value


def _success_rate(summary: dict) -> float:
    return float(summary["success_rate"])


def evaluate_phenomenon_gate(
    *, id_summary: dict, robotinit_summary: dict, thresholds: GateThresholds | None = None
) -> GateResult:
    t = thresholds or GateThresholds()
    gap_pp = (_success_rate(id_summary) - _success_rate(robotinit_summary)) * 100.0
    checks = {"distribution_shift_gap": gap_pp >= t.phenomenon_gap_pp}
    return GateResult(
        passed=all(checks.values()),
        checks=checks,
        values={"distribution_shift_gap_pp": gap_pp},
        thresholds=t,
        notes=[
            "G1 asks whether RobotInit is a strong enough failure mode to justify method development.",
            "If this gate fails, stop or change the backbone/problem instead of forcing a method.",
        ],
    )


def evaluate_method_gate(
    *,
    baseline_heldout: dict,
    method_heldout: dict,
    baseline_id: dict,
    method_id: dict,
    thresholds: GateThresholds | None = None,
) -> GateResult:
    t = thresholds or GateThresholds()
    heldout_gain_pp = (_success_rate(method_heldout) - _success_rate(baseline_heldout)) * 100.0
    id_drop_pp = (_success_rate(baseline_id) - _success_rate(method_id)) * 100.0
    baseline_tail = float(baseline_heldout.get("cvar20_pose_success", float("nan")))
    method_tail = float(method_heldout.get("cvar20_pose_success", float("nan")))
    tail_gain_pp = (method_tail - baseline_tail) * 100.0

    checks = {
        "heldout_gain": heldout_gain_pp >= t.min_heldout_gain_pp,
        "id_preservation": id_drop_pp <= t.max_id_drop_pp,
        "tail_gain": tail_gain_pp >= t.min_tail_gain_pp,
    }
    return GateResult(
        passed=all(checks.values()),
        checks=checks,
        values={
            "heldout_gain_pp": heldout_gain_pp,
            "id_drop_pp": id_drop_pp,
            "tail_gain_pp": tail_gain_pp,
        },
        thresholds=t,
        notes=[
            "The baseline should be the strongest matched-data baseline, normally targeted augmentation.",
            "Improving shifted performance by sacrificing clean competence does not pass.",
        ],
    )


def load_summary(path: str | Path) -> dict:
    value = read_json(path)
    if "success_rate" not in value:
        raise ValueError(f"{path} is not an evaluation summary.json")
    return value


def run_gate(
    *,
    mode: str,
    output: str | Path,
    id_summary: str | Path | None = None,
    robotinit_summary: str | Path | None = None,
    baseline_heldout: str | Path | None = None,
    method_heldout: str | Path | None = None,
    baseline_id: str | Path | None = None,
    method_id: str | Path | None = None,
) -> GateResult:
    if mode == "phenomenon":
        if id_summary is None or robotinit_summary is None:
            raise ValueError("phenomenon gate requires --id-summary and --robotinit-summary")
        result = evaluate_phenomenon_gate(
            id_summary=load_summary(id_summary),
            robotinit_summary=load_summary(robotinit_summary),
        )
    elif mode == "method":
        required = [baseline_heldout, method_heldout, baseline_id, method_id]
        if any(x is None for x in required):
            raise ValueError("method gate requires baseline/method heldout and ID summaries")
        result = evaluate_method_gate(
            baseline_heldout=load_summary(baseline_heldout),
            method_heldout=load_summary(method_heldout),
            baseline_id=load_summary(baseline_id),
            method_id=load_summary(method_id),
        )
    else:
        raise ValueError(f"Unknown gate mode: {mode}")

    write_json(result.to_dict(), output)
    return result
