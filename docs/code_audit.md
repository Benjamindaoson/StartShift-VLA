# Code and asset audit

This audit distinguishes implementation from scientific validation. **A** means an implemented utility can be run in the archive or a historical execution has saved evidence; it does not mean every deployment path is supported. **B** means partial implementation or unverified integration. **C** means design without completed implementation/experiment. **D** means deliberately retired from future development. D can also apply to an A/B component's maintenance status.

The original layout contained `startshift/`, `tests/`, `configs/`, `scripts/`, frozen `splits/`, split reports and an ignored `outputs/` tree. The archive retains package paths, promotes core output artifacts to `results/`, adds an executed-run specification and replaces the development README with an archive-oriented account. Earlier public designs remain in Git history. The [file inventory](source_inventory.md) covers retained Python, shell and configuration files; the [export manifest](../results/analysis/export_manifest.json) covers source-to-public provenance.

| Component and paths | Class | Implemented behavior and evidence | Incomplete or retired boundary |
|---|---|---|---|
| `startshift/cli.py`, `config.py`, `types.py`, `utils/io.py` | A | Argument parsing, configuration I/O and typed record structures; CLI help and unit tests | Presence of a subcommand does not prove its full integration |
| `data/classification.py`, `data/splits.py` | A | Classification normalization, deterministic split utilities, suite-qualified identity checks and adaptation-pool-only failure targeting; existing frozen files and tests | No physical-state disjointness proof; all manifest base-task labels absent; do not regenerate frozen files |
| `data/state.py` | A | State normalization and concatenation utilities; numeric tests | Eight-dimensional end-effector/gripper representation, not full joint-state coverage |
| `evaluation/metrics.py`, `evaluation/audit.py`, `gates.py` | A | Success, Wilson interval, worst-group/CVaR, descriptive gap and threshold reports | Historical G1 threshold pass is not causal identification |
| `evaluation/libero_id.py`, `evaluation/runner.py` | A, historical runtime | Canonical ID registration, policy loading, benchmark rollout and JSONL/summary export; 920 principal records | GPU simulation not rerun for archival; saved rows lack initial-state vectors and videos; failed ID setup is preserved |
| `analysis/aggregate.py`, `plots.py`, `report.py` | A utilities; B full study | Aggregation and report code; aggregation tests | No completed multi-seed RISE matrix for these utilities to analyze |
| `evaluation/failures.py` | B | Annotation-template and merge code | All 798 principal failures remain unannotated |
| `evaluation/matrix.py`, `evaluation/external.py` | B | Command plans and external evaluator hook; command-construction tests | No completed second-backbone or held-out method study |
| `diagnostics/state_coverage.py` | B | Initial-frame state extraction and distance computation code | No validated state-coverage study in the recorded results |
| `models/rise.py`, `models/checkpoint.py` | B; D development | RISE state projector, context branch and checkpoint helpers exist | No learned benefit, completed training or full-policy equivalence result; PyTorch tests require optional dependency |
| `training/augment.py`, `training/robust.py` | B; D development | Batch/inference transforms and GroupDRO computation | No end-to-end RISE evidence; GroupDRO tests are component tests only |
| `training/dataset.py` | B, known blocker | Initial-state cache and subset loader | Episode-level `task_index` missing in inspected metadata; bare IDs omit suite; official variant join absent |
| `training/baselines.py` | B | Explicit standard/expert/LoRA command flags and command tests | The dataset-selection path fails before the actual baseline training; no baseline efficacy result |
| `training/train.py` | B; D development | Optimizer loop, loss/logging and checkpoint code | No completed RISE training; inherited dataset mapping prevents valid matched-data use |
| `research_audit.py` | B metadata CLI; A tested pure utilities | Additive pre-archive metadata/split/summary audit functions with tests | CLI expects private Parquet metadata and original `outputs/` layout; archive users should use `scripts/verify_archive.py` instead |
| `scripts/verify_archive.py` | A | Offline source/public hash, record metric, seed and split verification | Saved-record verification only; no model or simulator import |
| `scripts/bootstrap.sh`, `download_libero_plus_assets.sh`, `env.sh`, `verify_environment.py` | D historical setup | Original dependency/assets/environment setup recipes retained for inspection | Not part of archive validation; external downloads and full simulation environment are unsupported |
| `scripts/run_m0.sh`, `resume_m0_id_gate.sh` | A historical / D active execution | Baseline launch history and reduced ID resume; principal records available | Broad M0 script regenerates splits; do not execute for archival reproduction |
| `scripts/run_full_matrix.py`, `run_eval_matrix.py`, `run_adaptation_example.sh`, `build_targeted_splits.sh` | B command recipes; D execution | Planning/launch helpers | No training matrix or additional evaluation launched during archival |
| `configs/`, `splits/`, `reports/` | A preserved assets | Core frozen partitions, evaluated subsets, G2-referenced budget subset and configuration recipes retained | Configurations are not results; planned 42/43/44 seeds were not all executed |
| Paired reset intervention, stronger system baseline, multiple policy validation, deployment study | C | Historical rationale only | No completed experiment; no continuation scheduled |

## Known source issues retained transparently

The episode selector and difficulty map use bare task IDs where variants are suite-local. The episode selector also expects a metadata field absent from the inspected converted dataset. These are material blockers, not merely missing documentation. Archival does not repair them by guessing a data join.

The runner seeds by record position, so changing group order changes rollout seeds. It records success if any timestep succeeds, while no actual initial state is written. Classification parsing can tolerate/skip some malformed input variants. These behaviors are preserved and constrain retrospective interpretation.

The model projector's local zero-initialized branch tests do not establish that a patched whole policy is equivalent under every preprocessing/padding path. Training configurations and code are retained as prototypes, without architecture changes or performance tuning.

## Packaging decisions

The principal baseline JSONL files and retained frozen split JSON files remain byte-identical to the private source. Summaries and logs with local paths are transformed in the public copy, with before/after hashes recorded. Credentials, migration reports, private operational Git history, dataset/model caches, checkpoints, environments and incomplete network downloads are excluded entirely. The original MIT attribution is retained; removing machine identity does not mean removing copyright attribution.

No new training, simulation, research-data download, model structure change or RISE optimization was performed for this archive. Detailed checks and any inherited code-quality failures appear in [verification](verification.md).
