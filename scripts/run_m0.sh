#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh

python scripts/verify_environment.py

startshift make-splits --output-dir splits --seed 42

startshift eval   --policy lerobot/smolvla_libero   --split splits/audit.json   --output-dir outputs/eval/m0/robotinit   --episodes 10   --method-name base   --record-trajectories

startshift eval-id   --policy lerobot/smolvla_libero   --output-dir outputs/eval/m0/id   --episodes 10   --method-name base

startshift audit   --robotinit outputs/eval/m0/robotinit/eval_records.jsonl   --id outputs/eval/m0/id/eval_records.jsonl   --output outputs/eval/m0/audit.json

startshift gate   --mode phenomenon   --id-summary outputs/eval/m0/id/summary.json   --robotinit-summary outputs/eval/m0/robotinit/summary.json   --output outputs/eval/m0/gate.json

echo
echo "M0 complete. Inspect outputs/eval/m0/gate.json before starting RISE training."
