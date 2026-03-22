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
RESUME_BASE_RUN_ID="${RESUME_BASE_RUN_ID:-}"
if [[ -n "${RESUME_BASE_RUN_ID}" ]]; then
  START_MODE="resume"
  BASE_RUN_ID="${RESUME_BASE_RUN_ID}"
fi
TARGET_RUN_IDS="${TARGET_RUN_IDS:-}"
if [[ -n "${TARGET_RUN_IDS}" && -z "${RESUME_BASE_RUN_ID}" && "${START_MODE}" == "fresh" ]]; then
  START_MODE="resume"
fi
SEEDS="${SEEDS:-}"
if [[ -z "${SEEDS}" && -n "${RESUME_BASE_RUN_ID}" && -f "${RESULTS_DIR}/runs/${RESUME_BASE_RUN_ID}/multi_seed_manifest.json" ]]; then
  SEEDS="$(RESULTS_DIR="${RESULTS_DIR}" RESUME_BASE_RUN_ID="${RESUME_BASE_RUN_ID}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import os

results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
run_id = os.environ.get("RESUME_BASE_RUN_ID", "").strip()
path = results_dir / "runs" / run_id / "multi_seed_manifest.json"
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = data.get("seeds") or []
    print(" ".join(str(int(s)) for s in seeds if str(s).strip() != ""))
except Exception:
    print("")
PY
)"
fi
SEEDS="${SEEDS:-42 43 44}"
WORKERS="${WORKERS:-2}"
EXP_WORKERS="${EXP_WORKERS:-2}"
N_ROUNDS="${N_ROUNDS:-}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-}"
if [[ -z "${N_ROUNDS}" && -n "${TRAIN_ROUNDS}" ]]; then
  if [[ "${TRAIN_ROUNDS}" =~ ^[0-9]+$ ]]; then
    N_ROUNDS="$((TRAIN_ROUNDS + 1))"
  else
    echo "错误：TRAIN_ROUNDS 需要是整数，当前=${TRAIN_ROUNDS}" 1>&2
    exit 4
  fi
fi
if [[ -n "${N_ROUNDS}" ]]; then
  if [[ ! "${N_ROUNDS}" =~ ^[0-9]+$ ]]; then
    echo "错误：N_ROUNDS 需要是整数，当前=${N_ROUNDS}" 1>&2
    exit 4
  fi
  if [[ "${N_ROUNDS}" -lt 2 ]]; then
    echo "错误：N_ROUNDS 必须 >= 2（最后一轮用于 test-only），例如 15轮训练+1轮测试请设 N_ROUNDS=16 或 TRAIN_ROUNDS=15" 1>&2
    exit 4
  fi
fi

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
elif [[ "${AB_TUNING}" == "A_only" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy
  )
elif [[ "${AB_TUNING}" == "A_matrix" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy
    full_model_A_lambda_policy_u_guardrail
    full_model_A_lambda_policy_ramp_guardrail
  )
elif [[ "${AB_TUNING}" == "A_matrix_plus_probe" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy
    full_model_A_lambda_policy_u_guardrail
    full_model_A_lambda_policy_ramp_guardrail
    full_model_A_lambda_policy_ramp_guardrail_train_probe
  )
elif [[ "${AB_TUNING}" == "A_ramp_guardrail_plus_probe_u_adaptive" ]]; then
  EXPERIMENTS=(
    full_model_A_lambda_policy_ramp_guardrail
    full_model_A_lambda_policy_ramp_guardrail_train_probe
    full_model_A_lambda_policy_ramp_guardrail_train_probe_u_adaptive
  )
elif [[ "${AB_TUNING}" == "baselines_only" ]]; then
  EXPERIMENTS=(
    baseline_random
    baseline_entropy
    baseline_coreset
    baseline_bald
    baseline_dial_style
    baseline_wang_style
  )
else
  EXPERIMENTS=(
    full_model_A_lambda_policy
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

if ! "${PYTHON_BIN}" - "${EXPERIMENTS[@]}" <<'PY'
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
then
  exit 3
fi

HAS_LLM_KEY="$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0,'src'); from config import Config; print('1' if bool(getattr(Config, 'LLM_API_KEY', None)) else '0')" 2>/dev/null || echo '0')"
NEED_LLM_KEY="0"
for exp in "${EXPERIMENTS[@]}"; do
  case "${exp}" in
    full_model_*)
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

if [[ -n "${TARGET_RUN_IDS}" ]]; then
  TARGET_RUN_IDS="${TARGET_RUN_IDS//,/ }"
  read -r -a TARGET_RUN_ID_ARR <<< "${TARGET_RUN_IDS}"
  if [[ "${#TARGET_RUN_ID_ARR[@]}" -eq 0 ]]; then
    echo "错误：TARGET_RUN_IDS 解析为空" 1>&2
    exit 4
  fi

  _run_target_run_id() {
    local rid="${1}"

    MERGED_EXPERIMENTS=()
    while IFS= read -r _exp_line; do
      _exp_line="$(echo "${_exp_line}" | xargs)"
      if [[ -n "${_exp_line}" ]]; then
        MERGED_EXPERIMENTS+=("${_exp_line}")
      fi
    done < <(RESULTS_DIR="${RESULTS_DIR}" RUN_ID="${rid}" "${PYTHON_BIN}" - "${EXPERIMENTS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
run_id = os.environ.get("RUN_ID", "").strip()
extras = [str(x).strip() for x in sys.argv[1:] if str(x).strip()]

existing = []
manifest_path = results_dir / "runs" / run_id / "manifest.json"
if manifest_path.exists():
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        exps = payload.get("experiments") if isinstance(payload, dict) else None
        if isinstance(exps, list):
            existing = [str(x).strip() for x in exps if str(x).strip()]
        elif isinstance(exps, dict):
            existing = [str(k).strip() for k in exps.keys() if str(k).strip()]
    except Exception:
        existing = []

merged = []
for x in existing + extras:
    if x and x not in merged:
        merged.append(x)

print("\n".join(merged))
PY
    )

    if [[ "${#MERGED_EXPERIMENTS[@]}" -eq 0 ]]; then
      echo "错误：无法为 run_id=${rid} 生成 experiments 列表" 1>&2
      exit 5
    fi

    if ! "${PYTHON_BIN}" - "${MERGED_EXPERIMENTS[@]}" <<'PY'
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
    then
      exit 3
    fi

    printf 'Running for run_id=%q (workers=%q exp_workers=%q):' "${rid}" "${WORKERS}" "${EXP_WORKERS}"
    printf ' %q' "${MERGED_EXPERIMENTS[@]}"
    printf '\n'

    "${PYTHON_BIN}" "src/experiments/run_all_experiments.py" \
      --results_dir "${RESULTS_DIR}" \
      --run_id "${rid}" \
      --start "resume" \
      --parallel_workers "${EXP_WORKERS}" \
      --experiments "${MERGED_EXPERIMENTS[@]}"
  }

  _WORKERS_I="${WORKERS}"
  if [[ -z "${_WORKERS_I}" ]]; then
    _WORKERS_I="1"
  fi
  if [[ "${_WORKERS_I}" -lt 1 ]]; then
    _WORKERS_I="1"
  fi

  _EXP_WORKERS_I="${EXP_WORKERS}"
  if [[ -z "${_EXP_WORKERS_I}" ]]; then
    _EXP_WORKERS_I="1"
  fi
  if [[ "${_EXP_WORKERS_I}" -lt 1 ]]; then
    _EXP_WORKERS_I="1"
  fi

  WORKERS="${_WORKERS_I}"
  EXP_WORKERS="${_EXP_WORKERS_I}"

  PIDS=()
  RID_NAMES=()
  FAIL=0

  for rid in "${TARGET_RUN_ID_ARR[@]}"; do
    rid="$(echo "${rid}" | xargs)"
    if [[ -z "${rid}" ]]; then
      continue
    fi

    if [[ "${WORKERS}" -le 1 ]]; then
      _run_target_run_id "${rid}"
      continue
    fi

    if [[ "${#PIDS[@]}" -ge "${WORKERS}" ]]; then
      pid0="${PIDS[0]}"
      rid0="${RID_NAMES[0]}"
      if ! wait "${pid0}"; then
        rc=$?
        echo "错误：run_id=${rid0} 运行失败 (exit=${rc})" 1>&2
        FAIL=1
      fi
      PIDS=("${PIDS[@]:1}")
      RID_NAMES=("${RID_NAMES[@]:1}")
    fi

    _run_target_run_id "${rid}" &
    PIDS+=("$!")
    RID_NAMES+=("${rid}")
  done

  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    rid="${RID_NAMES[$i]}"
    if ! wait "${pid}"; then
      rc=$?
      echo "错误：run_id=${rid} 运行失败 (exit=${rc})" 1>&2
      FAIL=1
    fi
  done

  if [[ "${FAIL}" -ne 0 ]]; then
    exit 6
  fi

  exit 0
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
