#![LOCAL_PATH] bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh

python scripts/verify_environment.py

if [[ ! -f splits/adapt_pool.json ]]; then
  startshift make-splits --output-dir splits --seed 42
fi

startshift eval   --policy lerobot/smolvla_libero   --split splits/adapt_pool.json   --output-dir outputs/eval/selection/base_adapt_pool   --episodes 10   --method-name base

startshift make-targeted   --pool splits/adapt_pool.json   --records outputs/eval/selection/base_adapt_pool/eval_records.jsonl   --output-dir splits   --budgets 10 25 50 100

echo "Failure-targeted split files written to splits/adapt_targeted_*.json"
