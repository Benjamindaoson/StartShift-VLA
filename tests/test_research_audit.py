from __future__ import annotations

import unittest

from startshift.research_audit import (
    audit_metadata_contract,
    audit_split_overlap,
    group_key,
    summarize_evaluations,
)


def record(suite="s", task_id=1, episode=0, success=True, seed=42):
    return dict(suite=suite, task_id=task_id, episode=episode, success=success,
                seed=seed, difficulty=1, initial_state=None, video=None)


class ResearchAuditTests(unittest.TestCase):
    def test_group_keys_are_suite_qualified(self):
        self.assertNotEqual(group_key({"suite": "a", "task_id": 7}),
                            group_key({"suite": "b", "task_id": 7}))

    def test_base_task_index_cannot_certify_robotinit_provenance(self):
        result = audit_metadata_contract(
            [{"episode_index": 0, "task_index": 7, "tasks": ["pick"]}],
            [{"task_index": 7, "task": "pick"}],
            [{"suite": "s", "task_id": 7}])
        self.assertEqual(result["status"], "BLOCKED_MISSING_VARIANT_PROVENANCE")
        self.assertEqual(result["verified_mapped_episodes"], 0)

    def test_missing_mapping_is_not_repaired_by_language(self):
        result = audit_metadata_contract(
            [{"episode_index": 0, "tasks": ["same_task_initstate_7"]}],
            [{"task_index": 0, "task": "same_task_initstate_7"}],
            [{"suite": "s", "task_id": 7}])
        self.assertFalse(result["ready_for_matched_data_training"])

    def test_duplicate_episode_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate episode"):
            audit_metadata_contract(
                [{"episode_index": 0}, {"episode_index": 0}], [], [])

    def test_observation_of_mapping_columns_is_not_official_verification(self):
        result = audit_metadata_contract(
            [{"episode_index": 0, "robotinit_suite": "s", "robotinit_task_id": 7}],
            [], [{"suite": "s", "task_id": 7}])
        self.assertEqual(result["status"], "UNVERIFIED_MAPPING_SOURCE")
        self.assertFalse(result["ready_for_matched_data_training"])

    def test_overlap_uses_suite_and_task_not_bare_id(self):
        splits = {"train": [{"suite": "a", "task_id": 7}],
                  "test": [{"suite": "b", "task_id": 7}]}
        self.assertEqual(audit_split_overlap(splits)["cross_split_overlaps"], {})
        splits["test"].append({"suite": "a", "task_id": 7})
        self.assertEqual(audit_split_overlap(splits)["cross_split_overlaps"],
                         {"train::test": ["a:7"]})

    def test_summary_recomputes_counts_and_observability(self):
        rows = [record(episode=0), record(episode=1, success=False)]
        result = summarize_evaluations(rows, bootstrap_replicates=20)
        self.assertEqual(result["n_episodes"], 2)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["missing_initial_states"], 2)
        self.assertEqual(result["missing_video_references"], 2)
        self.assertAlmostEqual(result["episode_wilson95_descriptive"][0],
                               0.09453120573423074)

    def test_duplicate_rollout_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate rollout"):
            summarize_evaluations([record(), record()], bootstrap_replicates=20)

    def test_string_success_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            summarize_evaluations([record(success="False")],
                                 bootstrap_replicates=20)

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [record(task_id=t, episode=i, success=(t == 1))
                for t in [1, 2] for i in [0, 1]]
        a = summarize_evaluations(rows, bootstrap_replicates=100, bootstrap_seed=7)
        b = summarize_evaluations(rows, bootstrap_replicates=100, bootstrap_seed=7)
        self.assertEqual(a["suite_stratified_pose_bootstrap95"],
                         b["suite_stratified_pose_bootstrap95"])


if __name__ == "__main__":
    unittest.main()
