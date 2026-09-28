# StartShift-VLA

**Diagnosing and Mitigating Hidden Reset-State Dependence in Vision-Language-Action Policies**

StartShift-VLA is a focused robot-learning project for studying **robot initial-state / reset-distribution shift** in VLA policies. The project asks a simple question:

> A policy already knows the task. If only the robot starts from a different valid configuration, does it still know how to act?

The project is intentionally **not** an agent framework, planner, world model, tactile stack, or multi-agent system. It is a VLA / robot-policy post-training and evaluation project.

---

## 1. Project thesis

Modern VLA policies consume images, language and proprioception, but training demonstrations still cover a narrow state manifold. A policy can therefore achieve high in-distribution task success while remaining dependent on the reset distribution used during data collection.

StartShift-VLA separates three possible responses to this failure mode:

1. **System solution** — restore/reposition to a familiar start condition.
2. **Data solution** — collect more diverse or targeted RobotInit demonstrations.
3. **Model solution** — adapt the policy with explicit initial-state conditioning.

The project does not assume in advance that the model solution wins. If targeted augmentation or reset/reposition solves the problem more cheaply, that is a valid result.

---

## 2. Core experimental question

Under the **same limited adaptation-data budget**, can explicit initial-state conditioning improve generalization to **held-out robot initial configurations** without degrading the original manipulation capability?

The core comparison is therefore matched-data:

- Base SmolVLA
- Standard fine-tuning
- Expert-only fine-tuning
- LoRA
- Random RobotInit augmentation
- Targeted hard-pose augmentation
- RISE-E
- RISE-EA
- optional RISE-EAR

The strongest simple baseline, not the base model, is the bar that RISE must beat.

---

## 3. Stack

- **Backbone:** `lerobot/smolvla_libero`
- **Framework:** Hugging Face LeRobot
- **ID benchmark:** LIBERO
- **Shift benchmark:** LIBERO-Plus, Robot Initial State dimension only
- **Primary policy input:** images + language + 8D LIBERO proprioception
- **Action:** 7D continuous end-effector delta + gripper
- **Candidate method:** RISE — Robust Initial-State Encoding and Adaptation
- **Optional physical validation:** SO-101, only after simulation gates pass

Pinned upstream revisions are recorded in [third_party.lock](third_party.lock).

---

## 4. RISE

### RISE-E: explicit initial-state encoding

The normal SmolVLA state input is the current normalized state (s_t). StartShift augments this representation with the episode initial state (s_0) and a normalized displacement term:

[
c_t = [s_t, s_0, s_t - s_0]
]

All three terms are retained:

- (s_t): where the robot is now
- (s_0): where the episode began
- (s_t-s_0): how far the body state has moved relative to the reset condition

In LIBERO this is an **8D end-effector/gripper state**, not a complete joint configuration.

### RISE-A: lightweight state adapter

RISE-A keeps the pretrained SmolVLA branch intact and learns only a zero-initialized context branch from ([s_0, s_t-s_0]). At initialization the patched model is behaviorally equivalent to the base policy. This is important: any gain after training can be attributed to learned reset-state context rather than a random architectural perturbation.

### RISE-R: tail-aware extension

`rise-ear` adds GroupDRO-style weighting over RobotInit difficulty groups. It is intentionally an extension, not the default method. Use it only after RISE-E/EA demonstrate that explicit state conditioning is useful and the remaining failure is concentrated in hard pose groups.

---

## 5. Metrics

The project does **not** optimize only mean success.

Every formal evaluation reports:

- ID success rate
- RobotInit mean success
- Wilson 95% confidence interval
- robustness retention = RobotInit SR / ID SR
- success by RobotInit difficulty
- success by perturbation pose
- worst-pose success
- lower-tail CVaR (worst 20% of poses)
- pose-success standard deviation
- adaptation budget
- trainable parameter count
- GPU hours / peak VRAM when available
- failure taxonomy

The central claim is only valid if it improves held-out RobotInit performance **and** preserves ID competence.

---

## 6. Hard project gates

The repository encodes explicit stop/go rules.

### G1 — phenomenon gate

RobotInit must be a material failure mode for the selected backbone.

Default internal threshold:

[
SR_{ID} - SR_{RobotInit} \ge 15\text{ pp}
]

If this fails, do not force RISE. Change the backbone/problem or stop.

### G2 — simple-solution gate

Run random and targeted RobotInit augmentation first. If a small amount of targeted data already closes the gap, the correct project conclusion may be about state/data coverage rather than a new adapter.

### G3 — method gate

Against the **strongest matched-data baseline**:

- held-out RobotInit gain >= 3 pp
- clean ID drop <= 2 pp
- lower-tail CVaR gain >= 3 pp

These are internal project gates, not universal research standards.

### G4 — generality

Repeat the phenomenon, and ideally the method trend, on a second policy/backbone before making a broad VLA claim.

### G5 — physical validation

Only after G0-G4 are credible should a real SO-101 experiment be added.

---

## 7. Repository layout

```text
startshift/
  data/           LIBERO-Plus metadata parsing, leak-free pose splits, state transforms
  diagnostics/    state coverage and pose-distance diagnostics
  models/         RISE projector/adapter and checkpoint format
  training/       dataset wrappers, baseline launchers, GroupDRO, custom RISE trainer
  evaluation/     rollout runner, audit, failure annotation, metrics
  analysis/       plots and Markdown report generation
  gates.py        explicit phenomenon/method stop-go rules
scripts/
  bootstrap.sh
  download_libero_plus_assets.sh
  run_m0.sh
  run_full_matrix.py
tests/
configs/
```

---

## 8. Installation

Formal experiments are Linux-only because LIBERO/LIBERO-Plus require MuJoCo.

```bash
git clone https://github.com/Benjamindaoson/StartShift-VLA.git
cd StartShift-VLA

bash scripts/bootstrap.sh
source scripts/env.sh
bash scripts/download_libero_plus_assets.sh
```

The bootstrap script pins the exact LeRobot and LIBERO-Plus commits in `third_party.lock`.

Before spending GPU time:

```bash
python scripts/verify_environment.py
pytest
ruff check startshift tests
```

---

## 9. Build the RobotInit manifest and splits

The upstream LIBERO-Plus task classification is normalized into stable StartShift records.

```bash
startshift manifest   --category robot   --output splits/robotinit_manifest.json

startshift make-splits   --output-dir splits   --seed 42   --adapt-budgets 10 25 50 100
```

Core splits are disjoint at the perturbation-group level:

- `audit.json`
- `adapt_pool.json`
- `dev.json`
- `heldout_test.json`

Nested adaptation files include:

- `adapt_random_10/25/50/100.json`
- `adapt_targeted_10/25/50/100.json`

**Do not randomly split frames.** A perturbation trajectory/group belongs to one core split only.

---

## 10. M0 — RobotInit audit

First reproduce the failure mode. Do not train RISE yet.

```bash
bash scripts/run_m0.sh
```

Or run explicitly:

```bash
startshift eval-id   --policy lerobot/smolvla_libero   --output-dir outputs/eval/m0/id   --episodes 10

startshift eval   --policy lerobot/smolvla_libero   --split splits/audit.json   --output-dir outputs/eval/m0/robotinit   --episodes 10   --record-trajectories

startshift audit   --id outputs/eval/m0/id/eval_records.jsonl   --robotinit outputs/eval/m0/robotinit/eval_records.jsonl   --output outputs/eval/m0/audit.json

startshift gate   --mode phenomenon   --id-summary outputs/eval/m0/id/summary.json   --robotinit-summary outputs/eval/m0/robotinit/summary.json   --output outputs/eval/m0/gate.json
```

If G1 fails, stop before method development.

---

## 11. Baselines

Print a reproducible command before executing it:

```bash
startshift baseline   --config configs/rise_ea.yaml   --split splits/adapt_targeted_25.json   --method lora   --output-dir outputs/train/lora_targeted_25
```

Execute with `--execute`.

Available baselines:

- `standard-ft`: unfreeze non-vision SmolVLA training path
- `expert-ft`: action expert + state projection
- `lora`: official LeRobot PEFT path

Random vs targeted augmentation are represented by using the corresponding split file with the same training recipe and budget.

---

## 12. Train RISE

RISE-E:

```bash
startshift train   --config configs/rise_e.yaml   --split splits/adapt_targeted_25.json   --method rise-e   --output-dir outputs/train/rise_e_b25
```

RISE-EA:

```bash
startshift train   --config configs/rise_ea.yaml   --split splits/adapt_targeted_25.json   --method rise-ea   --output-dir outputs/train/rise_ea_b25
```

Optional tail-aware extension:

```bash
startshift train   --config configs/rise_ear.yaml   --split splits/adapt_targeted_25.json   --method rise-ear   --output-dir outputs/train/rise_ear_b25
```

RISE checkpoints are intentionally lightweight: they store the base policy reference plus the learned state projector/adapter.

---

## 13. Held-out evaluation

Example:

```bash
startshift eval   --policy outputs/train/rise_ea_b25/last   --split splits/heldout_test.json   --output-dir outputs/eval/rise_ea_b25/heldout   --episodes 10   --method-name rise-ea   --adaptation-budget 25   --record-trajectories

startshift eval-id   --policy outputs/train/rise_ea_b25/last   --output-dir outputs/eval/rise_ea_b25/id   --episodes 10   --method-name rise-ea
```

For another LeRobot-compatible backbone, use `startshift external-eval` to generate/execute official `lerobot-eval` commands.

---

## 14. Failure annotation

Create a review sheet for failed rollouts:

```bash
startshift failure-template   --records outputs/eval/m0/robotinit/eval_records.jsonl   --output outputs/eval/m0/failures.csv
```

Labels:

- `TARGET_PERCEPTION`
- `GEOMETRY_ACTION_GROUNDING`
- `APPROACH`
- `GRASP`
- `CONTROL`
- `RECOVERY`
- `TIMEOUT`
- `OTHER`

Merge reviewed labels back:

```bash
startshift apply-failures   --records outputs/eval/m0/robotinit/eval_records.jsonl   --annotations outputs/eval/m0/failures.csv
```

---

## 15. Reports

After multiple methods are evaluated:

```bash
startshift report   --results-root outputs/eval   --output-dir reports/latest
```

The report generator produces the four project figures:

1. initial-state sensitivity
2. adaptation-data scaling
3. mean vs tail reliability
4. robustness vs trainable-parameter cost

---

## 16. Method gate

After evaluating the strongest matched-data baseline and RISE:

```bash
startshift gate   --mode method   --baseline-heldout outputs/eval/targeted_b25/heldout/summary.json   --method-heldout outputs/eval/rise_ea_b25/heldout/summary.json   --baseline-id outputs/eval/targeted_b25/id/summary.json   --method-id outputs/eval/rise_ea_b25/id/summary.json   --output outputs/eval/rise_ea_b25/gate.json
```

Use `--fail-on-reject` in automated experiment jobs if desired.

---

## 17. Full experiment plan

Generate the full seed × budget training plan:

```bash
python scripts/run_full_matrix.py
```

Execute only when the environment and splits have been verified:

```bash
python scripts/run_full_matrix.py --execute
```

Formal results should use multiple seeds and the same adaptation budget for competing methods.

---

## 18. What counts as a finished result

A result is not considered complete unless it has:

- pinned code/dataset revisions
- exact config and seed
- per-episode records
- mean + Wilson interval
- per-pose results
- worst-pose/CVaR metrics
- ID evaluation
- trainable parameter count
- GPU/time metadata
- failure examples/videos
- matched-data strongest baseline

Do not claim a RISE improvement from one seed, one task, or an unmatched data budget.

---

## 19. Scope exclusions

The first version deliberately excludes:

- high-level Agent planning
- RAG / tools / multi-agent
- world models
- tactile sensing
- HIL-SERL
- RTC
- large ROS2 runtime work
- mandatory SE(3) equivariance
- mandatory physical robot experiments

Those are separate research directions. StartShift-VLA stays focused on robot-policy state coverage, diagnosis and adaptation.

---

## 20. Status

**Code:** implemented.

**Still requires execution:** downloading upstream datasets/checkpoints, running GPU training/evaluation, manually reviewing failure videos, and producing empirical results. No repository can truthfully pre-compute those results without the required environment and GPU runs.

