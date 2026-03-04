#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-baseline_20260228_124857_seed42}"
SEED="${SEED:-42}"
EXECUTION="${EXECUTION:-parallel}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
NON_AGENT_WORKERS="${NON_AGENT_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

INCLUDE="${INCLUDE:-full_model_B_lambda_agent,fixed_lambda,random_lambda,rule_based_controller_r1,rule_based_controller_r2,rule_based_controller_r3,no_cold_start,fixed_k,no_normalization,no_agent,uncertainty_only,knowledge_only,agent_control_lambda,baseline_random,baseline_entropy,baseline_coreset,baseline_bald,baseline_dial_style,baseline_wang_style,baseline_llm_us,baseline_llm_rs}"

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
