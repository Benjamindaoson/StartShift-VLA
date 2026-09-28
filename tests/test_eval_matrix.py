from pathlib import Path

import yaml

from startshift.evaluation.matrix import build_plan


def test_eval_matrix_builds_heldout_and_id_commands(tmp_path: Path):
    config = {
        "heldout_split": "splits/heldout_test.json",
        "episodes": 3,
        "device": "cuda",
        "output_root": str(tmp_path / "eval"),
        "runs": [
            {
                "name": "rise-ea",
                "policy": "outputs/train/rise/last",
                "seed": 42,
                "adaptation_budget": 25,
                "trainable_parameters": 123,
            }
        ],
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = build_plan(path)
    assert len(plan) == 2
    assert plan[0][0:2] == ["startshift", "eval"]
    assert "--adaptation-budget" in plan[0]
    assert "25" in plan[0]
    assert plan[1][0:2] == ["startshift", "eval-id"]
