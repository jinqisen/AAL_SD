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

RUN_PATH="${RUN_PATH:-/Users/anykong/AD-KUCS/AAL_SD/results/runs/ab_geometry_v0_p3_20260327_151308}"
if [[ ! -d "${RUN_PATH}" ]]; then
  echo "错误：RUN_PATH 不存在：${RUN_PATH}" 1>&2
  exit 2
fi

RESULTS_DIR="${RESULTS_DIR:-results}"
BASE_RUN_ID="${BASE_RUN_ID:-$(basename "${RUN_PATH}")}"
SEEDS="${SEEDS:-}"
N_ROUNDS="${N_ROUNDS:-}"
START_MODE="${START_MODE:-fresh}"
WORKERS="${WORKERS:-3}"
EXP_WORKERS="${EXP_WORKERS:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-geometry_controller_v0}"
RESOLVE_ONLY="${1:-}"

eval "$(
  RUN_PATH="${RUN_PATH}" \
  BASE_RUN_ID="${BASE_RUN_ID}" \
  SEEDS="${SEEDS}" \
  N_ROUNDS="${N_ROUNDS}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import shlex
from pathlib import Path

run_path = Path(os.environ["RUN_PATH"]).resolve()
base_run_id = str(os.environ.get("BASE_RUN_ID", "")).strip() or run_path.name
seed_raw = str(os.environ.get("SEEDS", "")).strip()
n_rounds_raw = str(os.environ.get("N_ROUNDS", "")).strip()
experiment_name = str(os.environ.get("EXPERIMENT_NAME", "")).strip()

group_manifest_path = run_path / "multi_seed_manifest.json"
if not group_manifest_path.exists():
    raise SystemExit(f"missing multi_seed_manifest.json: {group_manifest_path}")

group_manifest = json.loads(group_manifest_path.read_text(encoding="utf-8"))
manifest_seeds = group_manifest.get("seeds")
if not isinstance(manifest_seeds, list) or not manifest_seeds:
    raise SystemExit(f"invalid seeds in {group_manifest_path}")

if seed_raw:
    seeds = [int(x) for x in seed_raw.replace(",", " ").split() if x.strip()]
else:
    seeds = [int(x) for x in manifest_seeds]

seed_n_rounds = []
for seed in seeds:
    seed_manifest_path = run_path.parent / f"{base_run_id}_seed{seed}" / "manifest.json"
    if not seed_manifest_path.exists():
        raise SystemExit(f"missing seed manifest: {seed_manifest_path}")
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    experiments = seed_manifest.get("experiments")
    if not isinstance(experiments, list) or experiment_name not in experiments:
        raise SystemExit(
            f"experiment {experiment_name} not found in {seed_manifest_path}"
        )
    cfg = seed_manifest.get("config")
    if isinstance(cfg, dict) and cfg.get("N_ROUNDS") is not None:
        seed_n_rounds.append(int(cfg["N_ROUNDS"]))

if n_rounds_raw:
    n_rounds = int(n_rounds_raw)
elif seed_n_rounds:
    n_rounds = seed_n_rounds[0]
    if any(x != n_rounds for x in seed_n_rounds):
        raise SystemExit(
            f"inconsistent N_ROUNDS across seeds: {seed_n_rounds}"
        )
else:
    n_rounds = 0

print(f"RESOLVED_BASE_RUN_ID={shlex.quote(base_run_id)}")
print(f"RESOLVED_SEEDS={shlex.quote(' '.join(str(x) for x in seeds))}")
print(f"RESOLVED_N_ROUNDS={shlex.quote(str(n_rounds))}")
PY
)"

if [[ "${RESOLVE_ONLY}" == "--resolve" ]]; then
  echo "RUN_PATH=${RUN_PATH}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "BASE_RUN_ID=${RESOLVED_BASE_RUN_ID}"
  echo "SEEDS=${RESOLVED_SEEDS}"
  echo "N_ROUNDS=${RESOLVED_N_ROUNDS}"
  echo "START_MODE=${START_MODE}"
  echo "WORKERS=${WORKERS}"
  echo "EXP_WORKERS=${EXP_WORKERS}"
  echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  exit 0
fi

cmd=(
  "${PYTHON_BIN}"
  "src/experiments/run_multi_seed.py"
  "--results_dir" "${RESULTS_DIR}"
  "--run_id" "${RESOLVED_BASE_RUN_ID}"
  "--seeds" "${RESOLVED_SEEDS}"
  "--start" "${START_MODE}"
  "--experiments" "${EXPERIMENT_NAME}"
  "--parallel"
  "--workers" "${WORKERS}"
  "--exp_workers" "${EXP_WORKERS}"
)

if [[ -n "${RESOLVED_N_ROUNDS}" && "${RESOLVED_N_ROUNDS}" != "0" ]]; then
  cmd+=("--n_rounds" "${RESOLVED_N_ROUNDS}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

RESULTS_DIR="${RESULTS_DIR}" \
BASE_RUN_ID="${RESOLVED_BASE_RUN_ID}" \
SEEDS="${RESOLVED_SEEDS}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from utils.multi_seed import aggregate_multi_seed, render_markdown

results_dir = Path(os.environ["RESULTS_DIR"]).resolve()
base_run_id = str(os.environ["BASE_RUN_ID"]).strip()
experiment_name = str(os.environ["EXPERIMENT_NAME"]).strip()
seeds = [int(x) for x in str(os.environ["SEEDS"]).replace(",", " ").split() if x.strip()]
group_dir = results_dir / "runs" / base_run_id
group_dir.mkdir(parents=True, exist_ok=True)
run_ids = []

for seed in seeds:
    run_id = f"{base_run_id}_seed{seed}"
    run_ids.append(run_id)
    run_dir = results_dir / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")

    merged = {}
    for result_path in sorted(run_dir.glob("result_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for exp_name, exp_payload in payload.items():
            if isinstance(exp_payload, dict):
                merged[str(exp_name)] = exp_payload

    if experiment_name not in merged:
        raise SystemExit(f"missing rerun result for {experiment_name} in {run_dir}")

    (run_dir / "experiment_results.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                manifest = payload
        except Exception:
            manifest = {}
    manifest["experiments"] = sorted(merged.keys())
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

group_manifest_path = group_dir / "multi_seed_manifest.json"
group_manifest = {}
if group_manifest_path.exists():
    try:
        payload = json.loads(group_manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            group_manifest = payload
    except Exception:
        group_manifest = {}
group_manifest["base_run_id"] = base_run_id
group_manifest["seeds"] = seeds
group_manifest["run_ids"] = run_ids
group_manifest["experiments"] = sorted(
    {
        exp_name
        for run_id in run_ids
        for exp_name in json.loads(
            (results_dir / "runs" / run_id / "experiment_results.json").read_text(encoding="utf-8")
        ).keys()
    }
)
group_manifest_path.write_text(
    json.dumps(group_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary = aggregate_multi_seed(results_dir, run_ids)
(group_dir / "multi_seed_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(group_dir / "multi_seed_report.md").write_text(
    render_markdown(summary),
    encoding="utf-8",
)
PY
