#!/usr/bin/env bash
set -euo pipefail

cd /Users/anykong/AD-KUCS/AAL_SD

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUNS_DIR="${RUNS_DIR:-${RESULTS_DIR}/runs}"
POOLS_DIR="${POOLS_DIR:-${RESULTS_DIR}/pools}"

SOURCE_BASE_RUN_ID="${SOURCE_BASE_RUN_ID:-baselines_only_p3_20260313_212023}"
BASE_RUN_ID="${BASE_RUN_ID:-ablation_matrix_p3_20260323_155500}"
SEEDS="${SEEDS:-42,43,44,45,46}"
START_MODE="${START_MODE:-fresh}"
N_ROUNDS="${N_ROUNDS:-16}"

EXPERIMENTS=(
  full_model_A_lambda_policy
  no_agent
  fixed_lambda
  uncertainty_only
  knowledge_only
)

"${PYTHON_BIN}" - "${EXPERIMENTS[@]}" <<'PY'
import sys
sys.path.insert(0, "src")
from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES

exps = sys.argv[1:]
resolved = [EXPERIMENT_NAME_ALIASES.get(e, e) for e in exps]
missing = [e for e, r in zip(exps, resolved) if r not in ABLATION_SETTINGS]
print("Resolved experiments:", " ".join(resolved))
if missing:
  raise SystemExit("Unknown experiments: " + " ".join(missing))
PY

python - <<PY
import os
seeds = "${SEEDS}".split(",")
src_base = "${SOURCE_BASE_RUN_ID}"
dst_base = "${BASE_RUN_ID}"
pools_dir = "${POOLS_DIR}"
for s in seeds:
  s = s.strip()
  if not s:
    continue
  src = os.path.join(pools_dir, f"{src_base}_seed{s}", "_base")
  dst = os.path.join(pools_dir, f"{dst_base}_seed{s}", "_base")
  print(f"{s}: {src} -> {dst}")
PY

for seed in ${SEEDS//,/ }; do
  SRC="${POOLS_DIR}/${SOURCE_BASE_RUN_ID}_seed${seed}/_base"
  DST="${POOLS_DIR}/${BASE_RUN_ID}_seed${seed}/_base"
  if [ ! -d "${SRC}" ]; then
    echo "Missing source pools: ${SRC}" 1>&2
    exit 2
  fi
  rm -rf "${DST}"
  mkdir -p "$(dirname "${DST}")"
  cp -a "${SRC}" "${DST}"
done

cmd=(
  "${PYTHON_BIN}"
  "src/experiments/run_multi_seed.py"
  "--results_dir" "${RESULTS_DIR}"
  "--run_id" "${BASE_RUN_ID}"
  "--seeds" "${SEEDS}"
  "--start" "${START_MODE}"
  "--n_rounds" "${N_ROUNDS}"
  "--experiments"
)
cmd+=("${EXPERIMENTS[@]}")

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
