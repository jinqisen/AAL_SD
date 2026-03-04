#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-baseline_20260228_124857_seed42}"
SEED="${SEED:-42}"
EXECUTION="${EXECUTION:-parallel}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
NON_AGENT_WORKERS="${NON_AGENT_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

RUN_DIR="${RUN_DIR:-results/runs/$RUN_ID}"
INCLUDE="${INCLUDE:-}"

if [[ -z "${INCLUDE}" && -d "${RUN_DIR}" ]]; then
  include_items=()
  shopt -s nullglob
  for status_path in "${RUN_DIR}"/*_status.json; do
    filename="$(basename "${status_path}")"
    exp_name="${filename%_status.json}"
    include_items+=("${exp_name}")
  done
  shopt -u nullglob

  if [[ ${#include_items[@]} -gt 0 ]]; then
    IFS=,
    INCLUDE="${include_items[*]}"
    unset IFS
  fi
fi

cmd=(python src/run_parallel_strict.py)
cmd+=(--resume "$RUN_ID")
cmd+=(--execution "$EXECUTION")
cmd+=(--seed "$SEED")
if [[ -n "${INCLUDE}" ]]; then
  cmd+=(--include "$INCLUDE")
fi
cmd+=(--agent-workers "$AGENT_WORKERS")
cmd+=(--non-agent-workers "$NON_AGENT_WORKERS")
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
"${cmd[@]}"
