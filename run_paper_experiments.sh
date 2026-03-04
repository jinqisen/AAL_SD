#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S_strict)}"
SEED="${SEED:-42}"
EXECUTION="${EXECUTION:-sequential}"
INCLUDE="${INCLUDE:-full_model,full_model_fixed_epochs_lambda_budget,agent_control_lambda,agent_control_budget,no_agent,uncertainty_only,knowledge_only,fixed_lambda,baseline_random,baseline_entropy,baseline_coreset,baseline_bald,baseline_llm_us,baseline_llm_rs}"
RESUME="${RESUME:-0}"
GENERATE_ASSETS="${GENERATE_ASSETS:-1}"
OVERWRITE="${OVERWRITE:-1}"
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

if [[ "$GENERATE_ASSETS" == "1" && "$DRY_RUN" != "1" ]]; then
  ASSET_ARGS=(--run_dir "results/runs/$RUN_ID")
  if [[ "${MULTI_SEED_GROUP_DIR:-}" != "" ]]; then
    ASSET_ARGS+=(--multi_seed_group_dir "$MULTI_SEED_GROUP_DIR")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    ASSET_ARGS+=(--overwrite)
  fi
  python paper/generate_paper_assets.py "${ASSET_ARGS[@]}"
fi

echo "$RUN_ID"
