#!/usr/bin/env bash
set -euo pipefail

AAL_SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AAL_SD_DIR}"

export AAL_SD_DATA_DIR="/Users/anykong/AAL_SD/data/Landslide4Sense"
export PYTHON_BIN="python"

RESULTS_DIR="results"
BASE_RUN_ID="aal_sd_full_model_k_prime_5seeds"
SEEDS="42 43 44 45 46"

# Run all seeds
for seed in $SEEDS; do
  echo "Starting seed $seed..."
  "${PYTHON_BIN}" src/main.py \
    --run_id "${BASE_RUN_ID}_seed${seed}" \
    --experiment_name "full_model_A_lambda_policy" \
    --start fresh \
    --n_rounds 16 &
done

wait
echo "All seeds finished."

# Aggregate results
"${PYTHON_BIN}" src/experiments/run_multi_seed.py \
  --results_dir "${RESULTS_DIR}" \
  --run_id "${BASE_RUN_ID}" \
  --seeds "42,43,44,45,46" \
  --start resume \
  --experiments full_model_A_lambda_policy

# Generate figures
"${PYTHON_BIN}" src/analysis/plot_paper_figures.py \
  --multi_seed_group_dir "${RESULTS_DIR}/runs/${BASE_RUN_ID}" \
  --output_dir AAL-SD-Doc/figures

