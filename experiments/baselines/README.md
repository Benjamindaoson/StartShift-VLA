# Recorded base-policy evaluation

`recorded_m0.json` describes the executed workshop baseline. The authoritative records are in `results/baseline/id/` and `results/baseline/robotinit/`. The failed ID launch and principal completion receipts are preserved in `results/historical/`. Preflight episodes and repetitive progress logs are omitted and do not contribute to principal metrics.

The historical rollout implementation is `startshift/evaluation/runner.py`; ID registration is in `startshift/evaluation/libero_id.py`. `scripts/resume_m0_id_gate.sh` records the reduced ID recipe. `scripts/run_m0.sh` is a broader historical recipe that regenerates split files and must not be used for archive verification.

No fine-tuning baseline or RISE method has a completed result in this archive.
