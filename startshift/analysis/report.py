from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from startshift.analysis.plots import (
    plot_data_scaling,
    plot_efficiency,
    plot_pose_success,
    plot_tail_reliability,
)
from startshift.types import EvalRecord
from startshift.utils.io import read_jsonl


def _discover(root: Path):
    for summary_path in root.rglob("summary.json"):
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        if "success_rate" in summary and "method" in summary:
            yield summary_path, summary


def build_report(results_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(results_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    pose_rows: list[dict] = []
    for summary_path, summary in _discover(root):
        row = dict(summary)
        metadata = row.get("metadata", {})
        row["adaptation_budget"] = metadata.get("adaptation_budget", 0)
        row["trainable_parameters"] = metadata.get("trainable_parameters", 0)
        row["gpu_hours"] = metadata.get("gpu_hours")
        summaries.append(row)

        records_path = summary_path.parent / "eval_records.jsonl"
        if records_path.exists():
            records = [EvalRecord.from_dict(x) for x in read_jsonl(records_path)]
            by_pose: dict[str, list[int]] = {}
            for record in records:
                by_pose.setdefault(record.pose_id or f"{record.suite}:{record.task_id}", []).append(int(record.success))
            for pose_id, values in by_pose.items():
                pose_rows.append(
                    {
                        "method": summary["method"],
                        "pose_id": pose_id,
                        "success_rate": sum(values) / len(values),
                    }
                )

    if not summaries:
        raise ValueError(f"No StartShift summary.json files found under {root}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "experiment_summary.csv", index=False)
    if pose_rows:
        pose_df = pd.DataFrame(pose_rows)
        pose_df.to_csv(out / "pose_success.csv", index=False)
        plot_pose_success(pose_df, out / "figure1_pose_sensitivity.png")

    if (summary_df["adaptation_budget"] > 0).any():
        plot_data_scaling(summary_df, out / "figure2_data_scaling.png")
    plot_tail_reliability(summary_df, out / "figure3_tail_reliability.png")
    if "trainable_parameters" in summary_df:
        plot_efficiency(summary_df, out / "figure4_efficiency.png")

    lines = [
        "# StartShift-VLA experiment report",
        "",
        f"Discovered {len(summary_df)} experiment summaries.",
        "",
        "## Aggregate results",
        "",
        summary_df[
            [
                "method",
                "success_rate",
                "worst_pose_success",
                "cvar20_pose_success",
                "pose_success_std",
                "adaptation_budget",
                "trainable_parameters",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation checklist",
        "",
        "- Does RobotInit substantially reduce performance relative to ID?",
        "- Does Targeted Augmentation solve the problem at the same data budget?",
        "- Does RISE beat the strongest matched-data baseline?",
        "- Is ID competence preserved?",
        "- Are worst-pose and CVaR gains consistent with the mean gain?",
        "- Are gains stable across seeds and a second policy/backbone?",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines))
    return report
