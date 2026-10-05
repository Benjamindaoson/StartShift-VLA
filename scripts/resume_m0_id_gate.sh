#![LOCAL_PATH] bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="${STARTSHIFT_POLICY_PATH:?Set STARTSHIFT_POLICY_PATH to an existing local policy snapshot}"
OUT=outputs/eval/m0_workshop
RECEIPTS=outputs/run_receipts
mkdir -p "$OUT/id" "$RECEIPTS"

start_epoch="$(date +%s)"
{
  printf 'stage=id_resume\n'
  printf 'start=%s\n' "$(date -Is)"
  printf 'start_epoch=%s\n' "$start_epoch"
  printf 'pid=%s\n' "$$"
  printf 'project_git_sha=%s\n' "$(git rev-parse HEAD)"
  sha256sum startshift/evaluation/libero_id.py startshift/evaluation/runner.py tests/test_libero_id.py
} > "$RECEIPTS/m0_id_resume_start.txt"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 30 > "$OUT/id_resume_nvidia.csv" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>[LOCAL_PATH] || true
  wait "$monitor_pid" 2>[LOCAL_PATH] || true
}
trap cleanup_monitor EXIT

printf 'stage=id_resume\nstatus=running\nupdated=%s\n' "$(date -Is)" > "$RECEIPTS/m0_workshop_stage.txt"
startshift eval-id \
  --policy "$MODEL" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --tasks-per-suite 3 \
  --episodes 10 \
  --output-dir "$OUT/id" \
  --seed 42 \
  --device cuda \
  --method-name base \
  > "$OUT/id.log" 2>&1
id_rc=$?
id_end_epoch="$(date +%s)"
{
  printf 'stage=id_resume\n'
  printf 'end=%s\n' "$(date -Is)"
  printf 'end_epoch=%s\n' "$id_end_epoch"
  printf 'elapsed_seconds=%s\n' "$((id_end_epoch - start_epoch))"
  printf 'rc=%s\n' "$id_rc"
} > "$RECEIPTS/m0_id_resume_end.txt"

if [[ "$id_rc" -ne 0 ]]; then
  printf 'stage=id_resume\nstatus=failed\nrc=%s\nupdated=%s\n' "$id_rc" "$(date -Is)" > "$RECEIPTS/m0_workshop_stage.txt"
  exit "$id_rc"
fi

printf 'stage=audit_gate\nstatus=running\nupdated=%s\n' "$(date -Is)" > "$RECEIPTS/m0_workshop_stage.txt"
audit_start_epoch="$(date +%s)"
startshift audit \
  --id "$OUT/id/eval_records.jsonl" \
  --robotinit "$OUT/robotinit/eval_records.jsonl" \
  --output "$OUT/audit.json" \
  > "$OUT/audit.log" 2>&1
audit_rc=$?
if [[ "$audit_rc" -eq 0 ]]; then
  startshift gate \
    --mode phenomenon \
    --id-summary "$OUT/id/summary.json" \
    --robotinit-summary "$OUT/robotinit/summary.json" \
    --output "$OUT/gate.json" \
    > "$OUT/gate.log" 2>&1
  gate_rc=$?
else
  gate_rc=99
fi
audit_end_epoch="$(date +%s)"
{
  printf 'stage=audit_gate\n'
  printf 'end=%s\n' "$(date -Is)"
  printf 'end_epoch=%s\n' "$audit_end_epoch"
  printf 'elapsed_seconds=%s\n' "$((audit_end_epoch - audit_start_epoch))"
  printf 'audit_rc=%s\n' "$audit_rc"
  printf 'gate_rc=%s\n' "$gate_rc"
} > "$RECEIPTS/m0_audit_gate_end.txt"

if [[ "$audit_rc" -ne 0 || "$gate_rc" -ne 0 ]]; then
  printf 'stage=audit_gate\nstatus=failed\naudit_rc=%s\ngate_rc=%s\nupdated=%s\n' \
    "$audit_rc" "$gate_rc" "$(date -Is)" > "$RECEIPTS/m0_workshop_stage.txt"
  exit 1
fi

printf 'stage=g1\nstatus=complete\nupdated=%s\n' "$(date -Is)" > "$RECEIPTS/m0_workshop_stage.txt"
