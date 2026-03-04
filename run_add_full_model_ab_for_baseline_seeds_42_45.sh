#!/usr/bin/env bash
set -euo pipefail

DEFAULT_RUN_IDS=(
  "baseline_20260228_124857_seed42"
  "baseline_20260228_124857_seed43"
  "baseline_20260228_124857_seed44"
  "baseline_20260228_124857_seed45"
)

RUN_IDS=("$@")
if (( ${#RUN_IDS[@]} == 0 )); then
  RUN_IDS=("${DEFAULT_RUN_IDS[@]}")
fi

INCLUDE="${INCLUDE:-full_model_A_lambda_policy,full_model_B_lambda_agent}"
EXECUTION="${EXECUTION:-parallel}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
NON_AGENT_WORKERS="${NON_AGENT_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

for RUN_ID in "${RUN_IDS[@]}"; do
  SEED="${SEED:-}"
  if [[ "$RUN_ID" =~ seed([0-9]+) ]]; then
    SEED="${BASH_REMATCH[1]}"
  fi
  if [[ "${SEED:-}" == "" ]]; then
    SEED="42"
  fi

  cmd=(python src/run_parallel_strict.py)
  cmd+=(--resume "$RUN_ID")
  cmd+=(--execution "$EXECUTION")
  cmd+=(--seed "$SEED")
  cmd+=(--include "$INCLUDE")
  cmd+=(--agent-workers "$AGENT_WORKERS")
  cmd+=(--non-agent-workers "$NON_AGENT_WORKERS")
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  "${cmd[@]}"
done
