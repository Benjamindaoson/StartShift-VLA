# Experiment design: executed and proposed

## Executed baseline

The base policy was `lerobot/smolvla_libero`, snapshot `31d453f7edd78c839a8bbc39744a292686daf0de`. Four suites were used: `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`.

- ID: the first three canonical tasks per suite, 10 episodes per task, 120 total.
- RobotInit: the frozen `audit_workshop_80.json` subset, 20 variant groups per suite and 10 episodes per group, 800 total. Each difficulty level contributes 160 episodes.
- Both used `hard_reset=true` and benchmark init states. Recorded success is whether the rollout reported success at any step; steps and accumulated rewards are retained in each JSONL row.
- Base seed 42. The runner assigns `42 + record_index * 10000 + episode_index`. These are per-rollout seeds from one schedule, not three independently replicated experiments.
- `startshift/evaluation/libero_id.py` installs canonical LIBERO task registration to distinguish the ID path from the LIBERO-Plus path. An initial ID attempt failed loading init-state assets; its failed receipt and traceback are preserved separately.

The saved rows are the authoritative execution evidence. `configs/m0_audit.yaml` and `scripts/run_m0.sh` describe a broader recipe and **are not an exact record of the reduced workshop run**. The archive-specific [executed-run manifest](../experiments/baselines/recorded_m0.json) records the observed settings. The original `resume_m0_id_gate.sh` documents the reduced ID invocation.

## Frozen partitions and data flow

Classification parsing produces suite-qualified RobotInit groups. Frozen audit, adaptation-pool, development and held-out JSON partitions remain unchanged. Workshop subsets refer to those parent partitions. Split identity checks compare `(suite, task_id)`; disjoint labels do not prove physically distinct simulator states.

All 1,550 manifest entries lack explicit `base_task`; stratification therefore actually operated over suite and difficulty rather than verified base-task identities. Adaptation budgets denote pose groups, not necessarily equal episode counts, frame exposure or optimization exposure.

Evaluation follows CLI → split/task records → policy/preprocessors → LeRobot rollout → episode JSONL → summary/gate. Analysis reads the saved records. Failure-annotation utilities exist but the principal failures were never annotated.

Training was intended to map selected pose groups to dataset episodes, attach the initial frame's state, normalize features, train and save a RISE context branch. The mapping failed before GPU training. The selector indexes episode rows by `task_index`, while the inspected metadata exposes task strings without a verified episode-to-RobotInit-variant join. Bare numeric task IDs are also suite-local and cannot safely serve as global variant identities. No such mapping was invented during archival.

## Unexecuted design

Historical configurations specify seeds 42/43/44, budgets 10/25/50/100 groups, 10,000 optimizer steps, batch size 8 and learning rate 1e-4. The intended comparisons include standard fine-tuning, expert fine-tuning, LoRA, random/targeted augmentation and RISE-E/EA/EAR. Their configuration files and launch-command tests are not evidence of completed training.

Original gates were: a descriptive ID–RobotInit gap of at least 15 percentage points; simple-data baselines first; then at least 3 points held-out and tail improvement over the strongest matched baseline with no more than 2 points ID regression; a second backbone; and eventual physical validation. Only the historical arithmetic G1 gate passed. It did not establish causal identification. G2 stopped at metadata validation; G3–G5 were not demonstrated.

Paired task/scene/reset intervention, multi-model verification and physical deployment were not executed. These are requirements for stronger successor research, not a new execution plan for this archive.
