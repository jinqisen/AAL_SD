#!/usr/bin/env bash
set -euo pipefail

AAL_SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AAL_SD_DIR}"

# Explicitly set dataset path (resolved from data的替身)
export AAL_SD_DATA_DIR="/Users/anykong/AAL_SD/data/Landslide4Sense"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  echo "错误：当前 PYTHON_BIN=${PYTHON_BIN} 无法 import torch" 1>&2
  echo "建议：" 1>&2
  echo "  1) 在已安装 torch 的环境中运行（例如 conda env），并指定 PYTHON_BIN=python" 1>&2
  echo "  2) 或者为该解释器安装 torch（pip/conda 安装对应版本）" 1>&2
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-results}"
START_MODE="${START_MODE:-fresh}"
BASE_RUN_ID="${BASE_RUN_ID:-baseline_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42 43 44 45}"
WORKERS="${WORKERS:-1}"
EXP_WORKERS="${EXP_WORKERS:-4}"
N_ROUNDS="${N_ROUNDS:-}"

AB_TUNING="${AB_TUNING:-}"
if [[ "${AB_TUNING}" == "lo" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy_ab_tune_lo
    full_model_B_lambda_agent_ab_tune_lo
  )
elif [[ "${AB_TUNING}" == "hi" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy_ab_tune_hi
    full_model_B_lambda_agent_ab_tune_hi
  )
elif [[ "${AB_TUNING}" == "both" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy_ab_tune_lo
    full_model_A_lambda_policy_ab_tune_hi
    full_model_B_lambda_agent_ab_tune_lo
    full_model_B_lambda_agent_ab_tune_hi
  )
else
  EXPERIMENTS=(
    full_model_A_lambda_policy
    full_model_A_lambda_policy_ab_tune_lo
    full_model_A_lambda_policy_ab_tune_hi
    full_model_A_lambda_policy_ab_tune_lo_ep10
    full_model_A_lambda_policy_ab_tune_hi_ep10
    full_model_B_lambda_agent
    baseline_random
    baseline_entropy
    baseline_coreset
    baseline_bald
    baseline_dial_style
    baseline_wang_style
  )
fi

HAS_LLM_KEY="$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0,'src'); from config import Config; print('1' if bool(getattr(Config, 'LLM_API_KEY', None)) else '0')" 2>/dev/null || echo '0')"
NEED_LLM_KEY="0"
for exp in "${EXPERIMENTS[@]}"; do
  case "${exp}" in
    full_model_A_lambda_policy|full_model_B_lambda_agent|full_model_A_lambda_policy_ab_tune_*|full_model_B_lambda_agent_ab_tune_*)
      NEED_LLM_KEY="1"
      break
      ;;
  esac
done
if [[ "${NEED_LLM_KEY}" == "1" && "${HAS_LLM_KEY}" != "1" ]]; then
  echo "错误：未检测到 LLM_API_KEY，但本脚本将运行 agent 实验（full_model_*）。" 1>&2
  echo "请配置 src/llm_config.json（或设置对应 API key 环境变量，例如 SILICONFLOW_API_KEY）后重试。" 1>&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}"
  "src/experiments/run_multi_seed.py"
  "--results_dir" "${RESULTS_DIR}"
  "--run_id" "${BASE_RUN_ID}"
  "--seeds" "${SEEDS}"
  "--start" "${START_MODE}"
  "--experiments" "${EXPERIMENTS[@]}"
  "--parallel"
  "--workers" "${WORKERS}"
  "--exp_workers" "${EXP_WORKERS}"
)

if [[ -n "${N_ROUNDS}" ]]; then
  cmd+=("--n_rounds" "${N_ROUNDS}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
