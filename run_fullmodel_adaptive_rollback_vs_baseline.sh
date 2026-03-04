#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S_rbth)}"
SEED="${SEED:-42}"
EXECUTION="${EXECUTION:-sequential}"
INCLUDE="${INCLUDE:-full_model_adaptive_rollback}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

MODE_ARGS=(--run-id "$RUN_ID")
if [[ "$RESUME" == "1" ]]; then
  MODE_ARGS=(--resume "$RUN_ID")
fi

EXTRA_ARGS=()
if [[ "${N_ROUNDS:-}" != "" ]]; then
  EXTRA_ARGS+=(--n-rounds "$N_ROUNDS")
fi
if [[ "${EPOCHS_PER_ROUND:-}" != "" ]]; then
  EXTRA_ARGS+=(--epochs-per-round "$EPOCHS_PER_ROUND")
fi

cmd=(python src/run_parallel_strict.py)
cmd+=("${MODE_ARGS[@]}")
cmd+=(--execution "$EXECUTION")
cmd+=(--seed "$SEED")
cmd+=(--include "$INCLUDE")
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  cmd+=("${EXTRA_ARGS[@]}")
fi
"${cmd[@]}"

echo "$RUN_ID"
