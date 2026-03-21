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
SEEDS="${SEEDS:-42 43 44}"
WORKERS="${WORKERS:-1}"
EXP_WORKERS="${EXP_WORKERS:-4}"
N_ROUNDS="${N_ROUNDS:-}"

EXPERIMENTS=(
  full_model_A_lambda_policy
  full_model_B_lambda_agent
  no_cold_start
  no_late_stage_ramp
  no_risk_modulation
  no_guardrail
  no_ema_smoothing
  no_cooling_period
  no_normalization
  no_agent
  uncertainty_only
  knowledge_only
  fixed_lambda
)

if ! "${PYTHON_BIN}" - "${EXPERIMENTS[@]}" <<'PY'; then
import sys
sys.path.insert(0, "src")
from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES

exps = sys.argv[1:]
missing = []
resolved = []
for e in exps:
  c = EXPERIMENT_NAME_ALIASES.get(e, e)
  resolved.append(c)
  if c not in ABLATION_SETTINGS:
    missing.append(e)
print("Resolved experiments:", " ".join(resolved))
if missing:
  print("错误：存在未知实验名：" + " ".join(missing), file=sys.stderr)
  raise SystemExit(3)
PY
  exit 3
fi

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
