#!/usr/bin/env bash
set -euo pipefail

cd /Users/anykong/AD-KUCS/AAL_SD

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src:.}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SOURCE_BASE_RUN_ID="${SOURCE_BASE_RUN_ID:-baselines_only_p3_20260313_212023}"
RUN_ID="${RUN_ID:-geometry_probe_p3_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42}"
START_MODE="${START_MODE:-fresh}"
N_ROUNDS="${N_ROUNDS:-16}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-2}"

cmd=(
  "env"
  "PYTHONPATH=${PYTHONPATH_VALUE}"
  "${PYTHON_BIN}"
  "src/experiments/run_geometry_probe_from_baselines.py"
  "--results_dir" "${RESULTS_DIR}"
  "--source_base_run_id" "${SOURCE_BASE_RUN_ID}"
  "--run_id" "${RUN_ID}"
  "--seeds" "${SEEDS}"
  "--start" "${START_MODE}"
  "--n_rounds" "${N_ROUNDS}"
  "--parallel_workers" "${PARALLEL_WORKERS}"
)

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
