#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh

BUDGET="${1:-25}"
SPLIT="splits/adapt_targeted_${BUDGET}.json"

startshift baseline --config configs/rise_ea.yaml --split "$SPLIT" \
  --method lora --output-dir "outputs/train/lora_${BUDGET}" --execute

startshift train --config configs/rise_e.yaml --split "$SPLIT" --method rise-e
startshift train --config configs/rise_ea.yaml --split "$SPLIT" --method rise-ea
