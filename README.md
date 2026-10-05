# StartShift-VLA: Diagnosing Hidden Reset-State Dependence in Vision-Language-Action Policies

**Research Status:** Archived research prototype. Further development redirected toward broader robot foundation model reliability research.

Does a robot vision-language-action (VLA) policy depend on the default reset state represented in its training demonstrations? StartShift-VLA investigated this question with SmolVLA, vanilla LIBERO and the Robot Initial States dimension of LIBERO-Plus.

This repository preserves the **core code, historical baseline records and research conclusions**. It is an honest research archive, not a validated mitigation method or a completed paper. RISE-E, RISE-EA and RISE-EAR remain unvalidated prototypes; their further development has stopped.

## What we observed

| Historical evaluation | Groups | Episodes | Successes | Success rate | Worst-group success | Lower-tail CVaR (20%) |
|---|---:|---:|---:|---:|---:|---:|
| ID initialization: vanilla LIBERO subset | 12 | 120 | 86 | **71.67%** | 30.00% | 46.67% |
| RobotInit shift: LIBERO-Plus audit subset | 80 | 800 | 36 | **4.50%** | 0.00% | 0.00% |

The observed gap is **67.17 percentage points**. Sixty-five of the 80 RobotInit groups recorded zero successes in ten trials. This is consistent with initialization sensitivity and identifies reset robustness as a concrete reliability concern.

**The comparison was not paired by base task and physical scene.** ID contains the first three canonical tasks per suite, while RobotInit contains a different selection of 80 variants. The full gap cannot be attributed to reset state alone. These results cover one policy and one seed schedule, and do not establish a general result for all VLA policies.

There is no validated RISE benefit, no completed matched-data training comparison, and no claim that this project solves initialization robustness. Training stopped at an unresolved episode-to-RobotInit-variant provenance check. The method effect is **N/A**, not zero.

## Verify the results in one command

Python 3.12+ is sufficient. No GPU, package installation, model weights, dataset downloads or network access are required after cloning:

```bash
git clone https://github.com/Benjamindaoson/StartShift-VLA.git
cd StartShift-VLA
python scripts/verify_archive.py
```

The verifier checks saved-file hashes, all **920 principal evaluation records**, success counts, historical summary metrics, frozen group identities and the recorded seed schedule. It recomputes arithmetic from existing records; it does not rerun the simulator.

To save a fresh summary without overwriting evidence:

```bash
python scripts/verify_archive.py --write-summary results/analysis/recomputed.json
```

## What is included

```text
startshift/              Core Python package: data, evaluation, analysis,
                        and explicitly unvalidated model/training prototypes
tests/                   Unit tests and archive-integrity tests
configs/                 Historical recipes; a config is not an executed result
splits/                  Core frozen partitions and evaluation/provenance subsets
experiments/             Recorded baseline settings and recipe navigation
results/baseline/        ID and RobotInit JSONL records and summaries
results/historical/      Essential failure log, gate, audit and run receipts
results/analysis/        Recomputed metrics and source-to-public hash manifest
scripts/                 Offline verifier and labeled historical research scripts
docs/                    Question, design, findings, limitations and reflection
LICENSE                  MIT license with original attribution
CITATION.cff             Software citation; no paper or DOI claimed
```

The public tree omits raw training datasets, videos, weights, checkpoints, GPU-monitor CSVs, repetitive progress logs, preflight retries, duplicate budget subsets and large generated viewers. Those operational files are unnecessary to inspect or verify the principal results. The complete preparation archive remains backed up privately. See [provenance and retained data](docs/archive_provenance.md).

## Implementation status

| Status | Scope |
|---|---|
| **A — implemented, with runnable or recorded evidence** | Offline metrics and audit utilities, split tools, historical base-policy evaluation |
| **B — partially implemented / integration unverified** | RISE models and trainer, training-data selector, failure annotation and second-backbone hooks |
| **C — designed, not completed** | Matched-data method study, multi-seed matrix, paired reset intervention and physical validation |
| **D — retired** | Active RISE development and automatic progression through the research matrix |

The [code audit](docs/code_audit.md) and [file inventory](docs/source_inventory.md) describe what exists and what remains incomplete. Core scientific code is preserved; it has not been optimized or restructured to imply completed research.

## Read the research record

- [Research question](docs/research_question.md) and [executed versus proposed experiment design](docs/experiment_design.md)
- [Results summary](results_summary.md): sample sizes, model, environment, metrics and execution evidence
- [Current findings](docs/findings.md) and [limitations](docs/limitations.md)
- [Research reflection](docs/research_reflection.md): why the project stops here and the direction toward **World Model Reliability / Reliable Physical Intelligence**

All 920 principal rows lack saved initial-state vectors and video references. All 798 failed episodes remain `UNLABELED`. A stricter study would need paired evaluation, multiple models, a failure taxonomy and deployment-oriented experiments. These are documented gaps, not new experiments scheduled in this archive.

## Optional package checks

Use a project-local environment for source checks:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-archive.txt -e ".[dev]"
python -m pytest tests/test_archive.py
python -m ruff check startshift tests scripts
python -m startshift.cli --help
```

The lightweight environment does not install PyTorch or LIBERO. The full historical suite was attempted: **41 passed, 2 PyTorch-dependent modules skipped, and 1 integration test failed because LIBERO was absent**. This is disclosed in [verification](docs/verification.md); no test was removed to claim full robotics validation. GitHub CI checks archive integrity and source quality only.

Historical setup, training and evaluation scripts can download resources, regenerate splits or launch experiments. They remain for inspection and are not the archive reproduction entry point. Use `scripts/verify_archive.py` for the recorded findings.

## License and citation

Project code retains its [MIT license](LICENSE). Third-party resources have their own licenses and are not bundled; see [third-party notices](docs/third_party.md).

Use [CITATION.cff](CITATION.cff) to cite the software and identify the exact commit used. No publication, DOI or method efficacy is claimed.
