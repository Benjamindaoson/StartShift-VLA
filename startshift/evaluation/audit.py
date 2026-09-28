from __future__ import annotations

from collections import Counter
from pathlib import Path

from startshift.evaluation.metrics import robustness_retention, summarize_records
from startshift.types import EvalRecord
from startshift.utils.io import read_jsonl, write_json


def load_eval_records(path: str | Path) -> list[EvalRecord]:
    return [EvalRecord.from_dict(row) for row in read_jsonl(path)]


def audit(
    robotinit_records: str | Path,
    *,
    output_path: str | Path,
    id_records: str | Path | None = None,
) -> dict:
    shifted = load_eval_records(robotinit_records)
    report = {"robotinit": summarize_records(shifted)}
    if id_records is not None:
        clean = load_eval_records(id_records)
        report["id"] = summarize_records(clean)
        report["robustness_retention"] = robustness_retention(
            report["robotinit"]["success_rate"], report["id"]["success_rate"]
        )
        report["id_regression_pp"] = (
            report["robotinit"]["success_rate"] - report["id"]["success_rate"]
        ) * 100.0

    failure_counts = Counter(r.failure_type or "UNLABELED" for r in shifted if not r.success)
    report["failure_counts"] = dict(failure_counts)
    report["failure_annotation_complete"] = all(
        r.success or (r.failure_type not in {None, "UNLABELED"}) for r in shifted
    )
    write_json(report, output_path)
    return report
