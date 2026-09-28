from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


CORE_METRICS = [
    "success_rate",
    "worst_pose_success",
    "cvar20_pose_success",
    "pose_success_std",
]


def discover_summaries(root: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in Path(root).rglob("summary.json"):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "method" not in value or "success_rate" not in value:
            continue
        metadata = value.get("metadata") or {}
        row = {
            "path": str(path),
            "method": value["method"],
            "success_rate": value.get("success_rate"),
            "worst_pose_success": value.get("worst_pose_success"),
            "cvar20_pose_success": value.get("cvar20_pose_success"),
            "pose_success_std": value.get("pose_success_std"),
            "adaptation_budget": metadata.get("adaptation_budget", 0),
            "seed": metadata.get("seed", value.get("seed")),
            "split": metadata.get("split"),
            "trainable_parameters": metadata.get("trainable_parameters", 0),
            "gpu_hours": metadata.get("gpu_hours"),
            "env_type": value.get("env_type"),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation summaries found under {root}")
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "adaptation_budget", "split"]
    available = [key for key in keys if key in df.columns]
    metrics = [metric for metric in CORE_METRICS if metric in df.columns]

    grouped = df.groupby(available, dropna=False)
    rows: list[dict] = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(available, group_key, strict=True))
        row["n_runs"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else math.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(available).reset_index(drop=True)


def matched_comparison(
    df: pd.DataFrame,
    *,
    baseline_method: str,
    method: str,
    metric: str = "success_rate",
) -> pd.DataFrame:
    """Compare methods only when adaptation budget and seed match.

    This intentionally refuses to average unmatched experiments because
    StartShift's central claim is a matched-data-budget claim.
    """
    required = {"method", "adaptation_budget", "seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for matched comparison: {sorted(missing)}")

    base = df[df["method"] == baseline_method][
        ["adaptation_budget", "seed", metric]
    ].rename(columns={metric: "baseline"})
    ours = df[df["method"] == method][
        ["adaptation_budget", "seed", metric]
    ].rename(columns={metric: "method_value"})
    merged = base.merge(ours, on=["adaptation_budget", "seed"], how="inner")
    if merged.empty:
        raise ValueError(
            f"No matched budget/seed runs for {baseline_method!r} vs {method!r}"
        )
    merged["gain"] = merged["method_value"] - merged["baseline"]
    merged["gain_pp"] = merged["gain"] * 100.0
    return merged.sort_values(["adaptation_budget", "seed"]).reset_index(drop=True)


def write_aggregate(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    baseline_method: str | None = None,
    method: str | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = discover_summaries(results_root)
    raw_path = out / "all_runs.csv"
    raw.to_csv(raw_path, index=False)

    aggregate = aggregate_seeds(raw)
    aggregate_path = out / "seed_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    result = {"all_runs": raw_path, "seed_aggregate": aggregate_path}
    if baseline_method and method:
        paired = matched_comparison(raw, baseline_method=baseline_method, method=method)
        paired_path = out / f"paired_{method}_vs_{baseline_method}.csv"
        paired.to_csv(paired_path, index=False)
        result["paired"] = paired_path
    return result
