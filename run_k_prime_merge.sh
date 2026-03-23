#!/usr/bin/env bash
set -euo pipefail

AAL_SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AAL_SD_DIR}"

export PYTHON_BIN="python"
RESULTS_DIR="results"
BASE_RUN_ID="baselines_only_p3_20260313_212023"
K_PRIME_RUN_ID="aal_sd_full_model_k_prime_5seeds"
SEEDS="42 43 44 45 46"

# Copy K prime results over when finished
for seed in $SEEDS; do
  echo "Copying K prime results for seed $seed..."
  mkdir -p "${RESULTS_DIR}/runs/${BASE_RUN_ID}_seed${seed}/full_model_A_lambda_policy_round_models"
  cp "${RESULTS_DIR}/runs/${K_PRIME_RUN_ID}_seed${seed}/result_full_model_A_lambda_policy.json" "${RESULTS_DIR}/runs/${BASE_RUN_ID}_seed${seed}/" 2>/dev/null || true
  cp "${RESULTS_DIR}/runs/${K_PRIME_RUN_ID}_seed${seed}/full_model_A_lambda_policy_trace.jsonl" "${RESULTS_DIR}/runs/${BASE_RUN_ID}_seed${seed}/" 2>/dev/null || true
done

echo "Aggregating multi-seed results for ${BASE_RUN_ID}..."
"${PYTHON_BIN}" src/experiments/run_multi_seed.py \
  --results_dir "${RESULTS_DIR}" \
  --run_id "${BASE_RUN_ID}" \
  --seeds "42,43,44,45,46" \
  --start resume \
  --experiments full_model_A_lambda_policy baseline_random baseline_entropy baseline_coreset baseline_bald baseline_dial_style baseline_wang_style

echo "Generating figures..."
"${PYTHON_BIN}" src/analysis/plot_paper_figures.py \
  --multi_seed_group_dir "${RESULTS_DIR}/runs/${BASE_RUN_ID}" \
  --output_dir AAL-SD-Doc/figures

