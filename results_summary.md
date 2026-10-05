# Results summary

**Status: archived research prototype.** All results below come from saved historical baseline records. Archive preparation ran analysis and software checks only; it launched no new training or simulator experiments.

## Completed principal evaluation

| Item | ID initialization | RobotInit shift |
|---|---|---|
| Environment | Vanilla LIBERO | LIBERO-Plus, Robot Initial States |
| Suites | Spatial, object, goal, long | Same four suite families |
| Selected groups | First 3 canonical tasks per suite: 12 | Frozen audit subset: 80 variants |
| Episodes per group | 10 | 10 |
| Total / successes / failures | 120 / 86 / 34 | 800 / 36 / 764 |
| Mean success | 71.67% | 4.50% |
| Episode-level Wilson 95% | 63.03–78.96% | 3.27–6.17% |
| Worst-group success | 30.00% | 0.00% |
| Lower-tail CVaR, worst 20% groups | 46.67% | 0.00% |
| Policy | SmolVLA, unchanged base checkpoint | Same checkpoint |
| Seed schedule | Base 42 plus 10,000 × group order + episode | Same rule, own group ordering |

Descriptive gap: **67.17 percentage points**. Ratio RobotInit/ID success: **6.28%**. Neither statistic is a paired causal estimate. Group ordering affects seeds, so equal base seed is not proof of paired trials.

The recorded environment was Python 3.12.14, PyTorch 2.11.0+cu130, CUDA runtime 13.0, Linux x86_64 and one NVIDIA GeForce RTX 4090 D. These hardware/software classes are retained for reproducibility; hostnames, IPs and machine-local paths are removed. LeRobot and LIBERO-Plus revisions are in [third_party.lock](third_party.lock), and model/dataset revisions are in [recorded_m0.json](experiments/baselines/recorded_m0.json).

The model revision is `31d453f7edd78c839a8bbc39744a292686daf0de`. The inspected training dataset revision is `f3f49f426d75030177b18778374005bc12ccd588`, with 14,347 episodes, 2,238,036 frames and 40 tasks. These are **available dataset metadata counts**, not adaptation examples consumed: the recorded G2 attempt started no training. The RobotInit classification manifest contains 1,550 groups; only the 80-group audit subset contributed the principal shifted result.

## Other preserved assets

- Preflight one-episode and ten-episode checks, retry progress logs and GPU telemetry are excluded from this core publication and from the 920-episode principal comparison. The complete preparation archive remains backed up privately. Principal completion/failure receipts remain under `results/historical/run_receipts`.
- The initial ID launch failed due to an init-state file lookup; its traceback and nonzero exit receipt are retained. A later ID run produced the 120 principal records.
- `results/historical/eval/m0_workshop/gate.json` preserves the historical G1 threshold outcome. Its `passed` flag means the arithmetic exceeded 15 points, not that causal identification or method validation passed.
- The G2 receipt explicitly records missing variant provenance and `gpu_training_started=false`. Baseline fine-tuning, RISE variants, multi-model and physical results are **N/A**.

## Interpretation and limitations

The observed difference is consistent with initialization sensitivity and makes reset robustness a credible reliability concern. Unequal task composition, one backbone, one seed schedule, missing initial states/videos and 798 unannotated failures limit interpretation. The archive contains no validated mitigation and makes no claim that the problem is solved.

The safe reproduction entry point is `python scripts/verify_archive.py`. It recomputes recorded metrics, checks seeds and split identities, and verifies exported evidence hashes. See [verification](docs/verification.md), [findings](docs/findings.md), [limitations](docs/limitations.md) and [research reflection](docs/research_reflection.md).
