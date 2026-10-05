import json

from startshift.data.classification import load_pose_records


def test_nested_libero_plus_schema(tmp_path):
    path = tmp_path / "task_classification.json"
    path.write_text(json.dumps({"libero_spatial":{"robot":{"L1":[1,2],"L3":[7]},"camera":{"L1":[3]}}}))
    records = load_pose_records(path, category="robot")
    assert [r.task_id for r in records] == [1, 2, 7]
    assert {str(r.difficulty) for r in records} == {"L1", "L3"}
    assert all(r.category == "robot" for r in records)


def test_current_libero_plus_schema_preserves_difficulty_level(tmp_path):
    path = tmp_path / "task_classification.json"
    path.write_text(
        json.dumps(
            {
                "libero_spatial": [
                    {
                        "id": 4,
                        "name": "pick_up_the_black_bowl_robot_init_4",
                        "category": "Robot Initial States",
                        "difficulty_level": 3,
                    }
                ]
            }
        )
    )
    records = load_pose_records(path, category="robot")
    assert len(records) == 1
    assert records[0].difficulty == 3
