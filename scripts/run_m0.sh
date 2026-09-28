#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh

startshift make-splits --output-dir splits --seed 42
startshift eval \
  --policy lerobot/smolvla_libero \
  --split splits/audit.json \
  --output-dir results/m0_robotinit \
  --episodes 10 \
  --method-name base

startshift eval-id \
  --policy lerobot/smolvla_libero \
  --output-dir results/m0_id \
  --episodes 10 \
  --method-name base

startshift audit \
  --robotinit results/m0_robotinit/eval_records.jsonl \
  --id results/m0_id/eval_records.jsonl \
  --output results/m0_audit.json
