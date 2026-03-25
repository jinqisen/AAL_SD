#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage:
  bash AAL_SD/resume_full_autotune.sh

Diagnostics:
  bash AAL_SD/resume_full_autotune.sh --resolve

Optional env overrides:
  BASELINE_RUN_ID   (default: autotune_opt_iter009_20260315_2014)
  BASELINE_EXP      (default: auto-detected best in BASELINE_RUN_ID)
  RUN_ID            (default: auto-detected latest autotune_opt_iter### run)
  SEEDS             (default: 42,43,44)
  BASE_SEED         (default: 42)
  CONFIRM_END_ROUND (default: 16)
  PROGRAM_PATH      (default: src/tuning_program.json)
  NO_LLM            (default: 0)  set to 1 to disable LLM proposer
USAGE
  exit 0
fi

BASELINE_RUN_ID="${BASELINE_RUN_ID:-autotune_opt_iter009_20260315_2014}"
SEEDS="${SEEDS:-42,43,44}"
BASE_SEED="${BASE_SEED:-42}"
BRANCH_ROUND="${BRANCH_ROUND:-7}"
SCREEN_OBJECTIVE="${SCREEN_OBJECTIVE:-alc}"
CONFIRM_OBJECTIVE="${CONFIRM_OBJECTIVE:-val}"
SCREEN_END_ROUND="${SCREEN_END_ROUND:-10}"
CONFIRM_END_ROUND="${CONFIRM_END_ROUND:-16}"
SCREEN_TOPK="${SCREEN_TOPK:-2}"
AGENT_WORKERS="${AGENT_WORKERS:-3}"
TARGET_MIOU="${TARGET_MIOU:-0.725}"
ORCH_MAX_ITERATIONS="${ORCH_MAX_ITERATIONS:-50}"
ORCH_SCREEN_EPOCHS_PER_ROUND="${ORCH_SCREEN_EPOCHS_PER_ROUND:-8}"
PROGRAM_PATH="${PROGRAM_PATH:-src/tuning_program.json}"
NO_LLM="${NO_LLM:-0}"
NO_AGENT="${NO_AGENT:-0}"
if [[ "${NO_AGENT}" == "1" ]]; then
  NO_LLM="1"
fi

if [[ -z "${BASELINE_EXP:-}" ]]; then
BASELINE_EXP="$(
  python - "${BASELINE_RUN_ID}" <<'PY'
import json
import re
from pathlib import Path
import os
import sys

baseline_arg = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
if not baseline_arg:
    raise SystemExit("missing BASELINE_RUN_ID")

run_dir = Path(baseline_arg)
if not run_dir.is_dir():
    run_dir = Path("results") / "runs" / baseline_arg
if not run_dir.exists():
    raise SystemExit(f"baseline run not found: {run_dir}")

re_last_val = re.compile(r"最后一轮选模 mIoU\(val\):\s*([0-9.]+)")
best_v = None
best_exp = None

for md in sorted(run_dir.glob("*.md")):
    try:
        txt = md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    vals = re_last_val.findall(txt)
    if not vals:
        continue
    try:
        v = float(vals[-1])
    except Exception:
        continue
    if best_v is None or v > best_v:
        best_v = v
        best_exp = md.stem

if not best_exp:
    for sp in sorted(run_dir.glob("*_status.json")):
        try:
            payload = json.loads(sp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if str(payload.get("status") or "").strip().lower() != "completed":
            continue
        res = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        v = res.get("final_mIoU")
        if not isinstance(v, (int, float)):
            continue
        if best_v is None or float(v) > float(best_v):
            best_v = float(v)
            best_exp = (
                str(payload.get("experiment_name") or "").strip()
                or sp.name.replace("_status.json", "")
            )

if not best_exp:
    raise SystemExit(f"unable to detect baseline exp from {run_dir}")
print(best_exp)
PY
)"
fi

if [[ -z "${RUN_ID:-}" ]]; then
RUN_ID="$(
  python - <<'PY'
import re
from pathlib import Path

runs_dir = Path("results") / "runs"
pat = re.compile(r"^autotune_opt_iter(\d+)_\d{8}_\d{4}$")

best_it = -1
best_name = None
for p in runs_dir.iterdir():
    if not p.is_dir():
        continue
    m = pat.match(p.name)
    if not m:
        continue
    it = int(m.group(1))
    if it > best_it:
        best_it = it
        best_name = p.name

if not best_name:
    raise SystemExit(f"no autotune_opt_iter runs found under {runs_dir}")
print(best_name)
PY
)"
fi

if [[ "${1:-}" == "--resolve" ]]; then
  echo "BASELINE_RUN_ID=${BASELINE_RUN_ID}"
  echo "BASELINE_EXP=${BASELINE_EXP}"
  echo "RUN_ID=${RUN_ID}"
  echo "SEEDS=${SEEDS}"
  echo "CONFIRM_END_ROUND=${CONFIRM_END_ROUND}"
  echo "NO_LLM=${NO_LLM}"
  exit 0
fi

cmd=(
  python src/resume_auto_tuning.py
  --run-id "${RUN_ID}"
  --baseline-run-id "${BASELINE_RUN_ID}"
  --baseline-exp "${BASELINE_EXP}"
  --seeds "${SEEDS}"
  --base-seed "${BASE_SEED}"
  --branch-round "${BRANCH_ROUND}"
  --screen-objective "${SCREEN_OBJECTIVE}"
  --confirm-objective "${CONFIRM_OBJECTIVE}"
  --screen-end-round "${SCREEN_END_ROUND}"
  --confirm-end-round "${CONFIRM_END_ROUND}"
  --screen-topk "${SCREEN_TOPK}"
  --agent-workers "${AGENT_WORKERS}"
  --target-miou "${TARGET_MIOU}"
  --orch-max-iterations "${ORCH_MAX_ITERATIONS}"
  --orch-screen-epochs-per-round "${ORCH_SCREEN_EPOCHS_PER_ROUND}"
  --program "${PROGRAM_PATH}"
  --start-monitor
  --continuous
)

if [[ "${NO_LLM}" == "1" ]]; then
  cmd+=(--no-llm)
fi

exec "${cmd[@]}"
