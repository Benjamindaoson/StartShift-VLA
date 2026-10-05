import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_archive", Path(__file__).parents[1] / "scripts/verify_archive.py"
)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def record(suite="s", task=1, episode=0, success=True, seed=42):
    return {"suite": suite, "task_id": task, "episode": episode,
            "success": success, "seed": seed, "failure_type": None if success else "UNLABELED"}


def test_duplicate_rollout_rejected():
    row = record()
    with pytest.raises(ValueError, match="Duplicate"):
        archive.summarize([row, copy.deepcopy(row)])


def test_same_local_task_id_in_different_suites_is_not_merged():
    result = archive.summarize([record(), record(suite="other", success=False)])
    assert result["groups"] == 2
    assert result["success_rate"] == 0.5
    assert result["zero_success_groups"] == 1


def test_string_boolean_rejected():
    with pytest.raises(ValueError, match="boolean"):
        archive.summarize([record(success="false")])


def test_empty_records_rejected():
    with pytest.raises(ValueError, match="Empty"):
        archive.summarize([])


def test_integrity_rejects_modified_file(tmp_path):
    p = tmp_path / "result.json"
    p.write_text("tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        archive.verify_hashes(tmp_path, [{"archive": "result.json", "public_sha256": "0" * 64}])


def test_integrity_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        archive.verify_hashes(tmp_path, [{"archive": "../outside.json", "public_sha256": "0" * 64}])
