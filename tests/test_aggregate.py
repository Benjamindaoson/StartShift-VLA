import pandas as pd
import pytest

from startshift.analysis.aggregate import aggregate_seeds, matched_comparison


def _frame():
    return pd.DataFrame(
        [
            {"method": "targeted", "adaptation_budget": 25, "seed": 1, "split": "heldout", "success_rate": 0.50, "cvar20_pose_success": 0.20},
            {"method": "targeted", "adaptation_budget": 25, "seed": 2, "split": "heldout", "success_rate": 0.52, "cvar20_pose_success": 0.22},
            {"method": "rise-ea", "adaptation_budget": 25, "seed": 1, "split": "heldout", "success_rate": 0.58, "cvar20_pose_success": 0.30},
            {"method": "rise-ea", "adaptation_budget": 25, "seed": 2, "split": "heldout", "success_rate": 0.57, "cvar20_pose_success": 0.29},
        ]
    )


def test_aggregate_seeds_reports_mean_and_run_count():
    result = aggregate_seeds(_frame())
    row = result[(result["method"] == "rise-ea") & (result["adaptation_budget"] == 25)].iloc[0]
    assert row["n_runs"] == 2
    assert row["success_rate_mean"] == pytest.approx(0.575)


def test_matched_comparison_is_paired_by_budget_and_seed():
    result = matched_comparison(_frame(), baseline_method="targeted", method="rise-ea")
    assert result["gain_pp"].tolist() == pytest.approx([8.0, 5.0])


def test_matched_comparison_rejects_missing_pairs():
    with pytest.raises(ValueError):
        matched_comparison(_frame(), baseline_method="missing", method="rise-ea")
