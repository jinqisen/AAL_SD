#!/usr/bin/env bash
set -euo pipefail

AAL_SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AAL_SD_DIR}"

export AAL_SD_DATA_DIR="${AAL_SD_DATA_DIR:-/Users/anykong/AAL_SD/data/Landslide4Sense}"

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
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-results}"
START_MODE="${START_MODE:-resume}"
if [[ -z "${BASE_RUN_ID:-}" ]]; then
  echo "错误：该脚本需要指定已有的 BASE_RUN_ID（与主结果相同）" 1>&2
  echo "示例：BASE_RUN_ID=baseline_20260309_123456 ./run_multiseed_ablation_controls_p4_resume.sh" 1>&2
  exit 2
fi
SEEDS="${SEEDS:-42 43 44 45}"
WORKERS="${WORKERS:-1}"
EXP_WORKERS="${EXP_WORKERS:-4}"
N_ROUNDS="${N_ROUNDS:-}"

EXPERIMENTS=(
  no_agent
  fixed_lambda
  random_lambda
  rule_based_controller_r1
  rule_based_controller_r2
  rule_based_controller_r3
  uncertainty_only
  knowledge_only
  no_cold_start
  fixed_k
  no_normalization
  bald_uncertainty
  agent_control_lambda
)

HAS_LLM_KEY="$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0,'src'); from config import Config; print('1' if bool(getattr(Config, 'LLM_API_KEY', None)) else '0')" 2>/dev/null || echo '0')"
if [[ "${HAS_LLM_KEY}" != "1" ]]; then
  echo "错误：未检测到 LLM_API_KEY，但本脚本包含 agent 实验（full_model_* / agent_control_lambda）。" 1>&2
  echo "请配置 src/llm_config.json（或设置对应 API key 环境变量，例如 SILICONFLOW_API_KEY）后重试。" 1>&2
  exit 3
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
