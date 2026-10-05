# Source inventory

Every retained source/script/configuration file is listed below. This is a navigation and implementation inventory; A does not imply full robotics integration. See `code_audit.md` for evidence and caveats. Entry points are extracted from the actual Python AST, not a presumed roadmap.

| File | Status | Top-level functions/classes or role |
|---|---|---|
| `configs/eval_matrix.example.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `configs/m0_audit.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `configs/matrix.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `configs/rise_e.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `configs/rise_ea.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `configs/rise_ear.yaml` | C planned recipe (M0 reduced execution documented separately) | Configuration |
| `scripts/bootstrap.sh` | D historical recipe | Historical shell recipe |
| `scripts/build_targeted_splits.sh` | D historical recipe | Historical shell recipe |
| `scripts/download_libero_plus_assets.sh` | D historical recipe | Historical shell recipe |
| `scripts/env.sh` | D historical recipe | Historical shell recipe |
| `scripts/resume_m0_id_gate.sh` | D historical recipe | Historical shell recipe |
| `scripts/run_adaptation_example.sh` | D historical recipe | Historical shell recipe |
| `scripts/run_eval_matrix.py` | D historical recipe | `main` |
| `scripts/run_full_matrix.py` | D historical recipe | `command`, `build_plan`, `main` |
| `scripts/run_m0.sh` | D historical recipe | Historical shell recipe |
| `scripts/verify_archive.py` | A utility / retained source | `summarize`, `verify_hashes`, `load_json`, `require`, `verify`, `main` |
| `scripts/verify_environment.py` | D historical recipe | `_git_head`, `_read_lock`, `_require_import`, `main` |
| `startshift/__init__.py` | A utility / retained source | Package initialization/constants |
| `startshift/analysis/__init__.py` | A utility / retained source | Package initialization/constants |
| `startshift/analysis/aggregate.py` | A utility / retained source | `discover_summaries`, `aggregate_seeds`, `matched_comparison`, `write_aggregate` |
| `startshift/analysis/plots.py` | A utility / retained source | `_save`, `plot_pose_success`, `plot_data_scaling`, `plot_tail_reliability`, `plot_efficiency` |
| `startshift/analysis/report.py` | A utility / retained source | `_discover`, `build_report` |
| `startshift/cli.py` | A utility / retained source | `_cmd_manifest`, `_cmd_splits`, `_cmd_make_targeted`, `_cmd_train`, `_cmd_baseline`, `_cmd_eval`, `_cmd_eval_id`, `_cmd_failure_template`, `_cmd_apply_failures`, `_cmd_audit`, `_cmd_aggregate`, `_cmd_report`, `_cmd_external_eval`, `_cmd_state_coverage`, `_cmd_gate`, `build_parser`, `main` |
| `startshift/config.py` | A utility / retained source | `DataConfig`, `PolicyConfig`, `TrainConfig`, `EvalConfig`, `ExperimentConfig`, `load_config`, `dump_config` |
| `startshift/constants.py` | A utility / retained source | Package initialization/constants |
| `startshift/data/__init__.py` | A utility / retained source | Package initialization/constants |
| `startshift/data/classification.py` | A utility / retained source | `normalize_category`, `find_classification_file`, `_looks_like_record`, `_get_first`, `_record_from_mapping`, `_walk_schema`, `load_pose_records` |
| `startshift/data/splits.py` | A utility / retained source | `_difficulty_rank`, `validate_disjoint_splits`, `build_pose_splits`, `save_pose_splits`, `load_pose_split`, `build_failure_targeted_split`, `save_failure_targeted_splits` |
| `startshift/data/state.py` | A utility / retained source | `MeanStdStats`, `augment_normalized_state`, `augment_normalized_state_torch`, `normalize_with_dataset_stats` |
| `startshift/diagnostics/__init__.py` | B integration/study incomplete | Package initialization/constants |
| `startshift/diagnostics/state_coverage.py` | B integration/study incomplete | `extract_task_initial_states`, `pose_distance_table`, `write_pose_distance_csv` |
| `startshift/evaluation/__init__.py` | A utility / retained source | Package initialization/constants |
| `startshift/evaluation/audit.py` | A utility / retained source | `load_eval_records`, `audit` |
| `startshift/evaluation/external.py` | B integration/study incomplete | `official_eval_args`, `run_official_eval` |
| `startshift/evaluation/failures.py` | B integration/study incomplete | `FailureType`, `write_annotation_template`, `apply_annotations` |
| `startshift/evaluation/libero_id.py` | A utility / retained source | `_VanillaLiberoSuite`, `install_vanilla_libero_benchmarks` |
| `startshift/evaluation/matrix.py` | B integration/study incomplete | `_cmd`, `build_plan` |
| `startshift/evaluation/metrics.py` | A utility / retained source | `wilson_interval`, `lower_tail_cvar`, `_group_success`, `summarize_records`, `robustness_retention`, `adaptation_efficiency` |
| `startshift/evaluation/runner.py` | A utility / retained source | `_load_policy`, `_episode_steps`, `evaluate_records`, `evaluate_split`, `evaluate_id_suites` |
| `startshift/gates.py` | A utility / retained source | `GateThresholds`, `GateResult`, `_success_rate`, `evaluate_phenomenon_gate`, `evaluate_method_gate`, `load_summary`, `run_gate` |
| `startshift/models/__init__.py` | B prototype; D development | Package initialization/constants |
| `startshift/models/checkpoint.py` | B prototype; D development | `save_rise_checkpoint`, `load_rise_checkpoint` |
| `startshift/models/rise.py` | B prototype; D development | `RISEProjectorConfig`, `RISEStateProjector`, `patch_smolvla_state_projector`, `freeze_all_except_rise`, `count_parameters` |
| `startshift/research_audit.py` | B integration/study incomplete | `group_key`, `audit_metadata_contract`, `audit_split_overlap`, `_wilson`, `_percentile`, `summarize_evaluations`, `_read_json`, `main` |
| `startshift/training/__init__.py` | B prototype; D development | `__getattr__` |
| `startshift/training/augment.py` | B prototype; D development | `RISEBatchTransform`, `StartShiftInferencePreprocessor` |
| `startshift/training/baselines.py` | B prototype; D development | `resolve_episode_subset`, `lerobot_train_args`, `run_baseline`, `shell_command` |
| `startshift/training/dataset.py` | B prototype; D development | `episodes_for_task_ids`, `difficulty_group_map`, `build_initial_state_cache`, `InitialStateDataset`, `load_subset_for_policy` |
| `startshift/training/robust.py` | B prototype; D development | `GroupDRO` |
| `startshift/training/train.py` | B prototype; D development | `_autocast`, `_make_policy`, `train_rise` |
| `startshift/types.py` | A utility / retained source | `PoseRecord`, `EvalRecord` |
| `startshift/utils/__init__.py` | A utility / retained source | Package initialization/constants |
| `startshift/utils/io.py` | A utility / retained source | `read_json`, `write_json`, `read_jsonl`, `write_jsonl` |
| `startshift/utils/repro.py` | A utility / retained source | `_git_sha`, `collect_environment`, `seed_everything` |
| `tests/test_aggregate.py` | A test source; scope in verification.md | `_frame`, `test_aggregate_seeds_reports_mean_and_run_count`, `test_matched_comparison_is_paired_by_budget_and_seed`, `test_matched_comparison_rejects_missing_pairs` |
| `tests/test_archive.py` | A test source; scope in verification.md | `record`, `test_duplicate_rollout_rejected`, `test_same_local_task_id_in_different_suites_is_not_merged`, `test_string_boolean_rejected`, `test_empty_records_rejected`, `test_integrity_rejects_modified_file`, `test_integrity_rejects_path_escape` |
| `tests/test_baselines.py` | A test source; scope in verification.md | `_args`, `test_standard_and_expert_ft_are_not_identical`, `test_lora_uses_official_top_level_peft_flags` |
| `tests/test_classification.py` | A test source; scope in verification.md | `test_nested_libero_plus_schema`, `test_current_libero_plus_schema_preserves_difficulty_level` |
| `tests/test_cli.py` | A test source; scope in verification.md | `test_cli_has_full_pipeline_commands` |
| `tests/test_config.py` | A test source; scope in verification.md | `test_config_round_trip` |
| `tests/test_eval_matrix.py` | A test source; scope in verification.md | `test_eval_matrix_builds_heldout_and_id_commands` |
| `tests/test_gates.py` | A test source; scope in verification.md | `_summary`, `test_phenomenon_gate`, `test_phenomenon_gate_rejects_gap_below_fifteen_percentage_points`, `test_method_gate_requires_gain_id_preservation_and_tail`, `test_method_gate_rejects_id_tradeoff` |
| `tests/test_groupdro.py` | A test source; scope in verification.md | `test_groupdro_upweights_hard_group` |
| `tests/test_libero_id.py` | A test source; scope in verification.md | `test_vanilla_libero_suite_overlay_restores_canonical_tasks` |
| `tests/test_metrics.py` | A test source; scope in verification.md | `test_lower_tail_cvar`, `test_summary_reports_mean_and_tail`, `test_wilson_interval_is_bounded` |
| `tests/test_research_audit.py` | A test source; scope in verification.md | `record`, `ResearchAuditTests` |
| `tests/test_rise.py` | A test source; scope in verification.md | `test_rise_linear_is_zero_init_equivalent_to_masked_base`, `test_rise_adapter_starts_equivalent_and_can_learn_context` |
| `tests/test_splits.py` | A test source; scope in verification.md | `_records`, `test_pose_splits_are_reproducible_and_disjoint`, `test_core_splits_preserve_uneven_strata`, `test_disjoint_validator_checks_adaptation_pool`, `test_adaptation_budgets_are_nested_and_difficulty_split_is_hard_first`, `test_failure_targeted_split_uses_empirical_failure_rate`, `test_failure_targeted_split_rejects_test_leakage` |
| `tests/test_state.py` | A test source; scope in verification.md | `test_state_representation_preserves_absolute_initial_and_relative`, `test_mean_std_round_trip` |
