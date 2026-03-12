import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_json(path: Path) -> Optional[Dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_source_perf_from_checkpoint(ckpt_payload: Dict) -> List[Dict]:
    ph = ckpt_payload.get("performance_history")
    if isinstance(ph, list):
        out = [r for r in ph if isinstance(r, dict)]
        out.sort(key=lambda r: int(r.get("round", 0) or 0))
        return out
    return []


def _load_source_perf_from_result_json(result_payload: Dict, exp_name: str) -> List[Dict]:
    section = result_payload.get(exp_name) if isinstance(result_payload, dict) else None
    if not isinstance(section, dict):
        return []
    ph = section.get("performance_history")
    if isinstance(ph, list):
        out = [r for r in ph if isinstance(r, dict)]
        out.sort(key=lambda r: int(r.get("round", 0) or 0))
        return out
    return []


def _get_perf_entry(perf: List[Dict], round_idx: int) -> Optional[Dict]:
    for r in perf:
        try:
            if int(r.get("round", 0) or 0) == int(round_idx):
                return r
        except Exception:
            continue
    return None


def _resolve_source_perf(
    runs_dir: Path, checkpoints_dir: Path, run_id: str, source_exp: str
) -> Tuple[List[Dict], str]:
    ckpt_path = checkpoints_dir / run_id / f"{source_exp}_state.json"
    ckpt_payload = _read_json(ckpt_path)
    if isinstance(ckpt_payload, dict):
        perf = _load_source_perf_from_checkpoint(ckpt_payload)
        if perf:
            return perf, str(ckpt_path)

    result_path = runs_dir / run_id / f"result_{source_exp}.json"
    result_payload = _read_json(result_path)
    if isinstance(result_payload, dict):
        perf = _load_source_perf_from_result_json(result_payload, source_exp)
        if perf:
            return perf, str(result_path)

    raise RuntimeError(
        f"Unable to find performance_history for source_exp={source_exp} in "
        f"{ckpt_path} or {result_path}"
    )


def _split_pools_at_labeled_size(
    pools_dir: Path, labeled_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labeled_path = pools_dir / "labeled_pool.csv"
    unlabeled_path = pools_dir / "unlabeled_pool.csv"
    if not labeled_path.exists() or not unlabeled_path.exists():
        raise FileNotFoundError(f"Missing pool csvs in {pools_dir}")

    labeled_df = pd.read_csv(labeled_path)
    unlabeled_df = pd.read_csv(unlabeled_path)

    labeled_size = int(labeled_size)
    if labeled_size < 0:
        raise ValueError(f"Invalid labeled_size={labeled_size}")
    if int(len(labeled_df)) < labeled_size:
        raise ValueError(
            f"labeled_pool.csv too small: have {len(labeled_df)} < need {labeled_size}"
        )

    labeled_head = labeled_df.iloc[:labeled_size].copy()
    labeled_tail = labeled_df.iloc[labeled_size:].copy()
    unlabeled_at_cut = pd.concat([unlabeled_df, labeled_tail], ignore_index=True)
    return labeled_head, unlabeled_at_cut


def _write_pools(target_dir: Path, labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    labeled_tmp = target_dir / "labeled_pool.csv.tmp"
    unlabeled_tmp = target_dir / "unlabeled_pool.csv.tmp"
    labeled_df.to_csv(labeled_tmp, index=False)
    unlabeled_df.to_csv(unlabeled_tmp, index=False)
    os.replace(labeled_tmp, target_dir / "labeled_pool.csv")
    os.replace(unlabeled_tmp, target_dir / "unlabeled_pool.csv")


def _build_checkpoint_payload(
    target_exp: str,
    perf_upto: List[Dict],
    resume_round: int,
    labeled_size: int,
    unlabeled_size: int,
) -> Dict:
    budget_history: List[int] = []
    for r in perf_upto:
        try:
            budget_history.append(int(r.get("labeled_size")))
        except Exception:
            budget_history.append(0)
    budget_history = budget_history[: int(resume_round)]
    return {
        "round": int(resume_round),
        "performance_history": list(perf_upto),
        "budget_history": budget_history,
        "labeled_size": int(labeled_size),
        "unlabeled_size": int(unlabeled_size),
        "rng_states": {},
        "model_selection": "best_val",
        "experiment_name": str(target_exp),
    }


def _launch(
    *,
    python_bin: str,
    main_path: str,
    run_id: str,
    experiments: List[str],
    start_mode: str,
    data_dir: Optional[str],
    random_seed: Optional[int],
    deterministic: Optional[bool],
    max_concurrent: int,
    dry_run: bool,
) -> None:
    start_mode = str(start_mode or "resume").strip().lower()
    if start_mode not in ("resume", "fresh"):
        raise ValueError(f"Invalid start_mode={start_mode}")

    max_concurrent = int(max_concurrent or 1)
    if max_concurrent <= 0:
        max_concurrent = 1

    env_base = os.environ.copy()
    if data_dir:
        env_base["AAL_SD_DATA_DIR"] = str(data_dir)
    if random_seed is not None:
        env_base["AAL_SD_RANDOM_SEED"] = str(int(random_seed))
    if deterministic is not None:
        env_base["AAL_SD_DETERMINISTIC"] = "1" if bool(deterministic) else "0"

    procs: List[subprocess.Popen] = []
    pending = list(experiments)

    def _spawn(exp_name: str) -> subprocess.Popen:
        cmd = [
            str(python_bin),
            str(main_path),
            "--experiment_name",
            str(exp_name),
            "--run_id",
            str(run_id),
            "--start",
            str(start_mode),
        ]
        printable = " ".join([subprocess.list2cmdline([c]) for c in cmd])
        if dry_run:
            print(printable)
            return subprocess.Popen([str(python_bin), "-c", "print('dry_run')"])
        print(printable)
        return subprocess.Popen(cmd, env=env_base)

    while pending or procs:
        while pending and len(procs) < max_concurrent:
            exp = pending.pop(0)
            procs.append(_spawn(exp))

        alive: List[subprocess.Popen] = []
        for p in procs:
            rc = p.poll()
            if rc is None:
                alive.append(p)
            else:
                if not dry_run and rc != 0:
                    raise RuntimeError(f"Experiment process exited with code {rc}")
        procs = alive
        if pending and procs:
            try:
                procs[0].wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, type=str)
    parser.add_argument("--source_exp", required=True, type=str)
    parser.add_argument("--target_exps", required=True, type=str)
    parser.add_argument("--start_round", required=True, type=int)
    parser.add_argument("--pools_dir", default="results/pools", type=str)
    parser.add_argument("--checkpoints_dir", default="results/checkpoints", type=str)
    parser.add_argument("--runs_dir", default="results/runs", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch_python", default=None, type=str)
    parser.add_argument("--launch_main", default="src/main.py", type=str)
    parser.add_argument("--launch_start", default="resume", type=str)
    parser.add_argument("--launch_data_dir", default=None, type=str)
    parser.add_argument("--launch_seed", default=None, type=int)
    parser.add_argument("--launch_deterministic", default=None, type=int)
    parser.add_argument("--launch_max_concurrent", default=1, type=int)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id).strip()
    source_exp = str(args.source_exp).strip()
    target_exps = [x.strip() for x in str(args.target_exps).split(",") if x.strip()]
    if not target_exps:
        raise SystemExit("target_exps is empty")

    start_round = int(args.start_round)
    if start_round <= 1:
        raise SystemExit("start_round must be >= 2")
    resume_round = int(start_round - 1)

    pools_dir = Path(str(args.pools_dir)).expanduser().resolve()
    checkpoints_dir = Path(str(args.checkpoints_dir)).expanduser().resolve()
    runs_dir = Path(str(args.runs_dir)).expanduser().resolve()

    perf, perf_src = _resolve_source_perf(runs_dir, checkpoints_dir, run_id, source_exp)
    cut_entry = _get_perf_entry(perf, resume_round)
    if not isinstance(cut_entry, dict):
        raise SystemExit(
            f"Missing performance entry for resume_round={resume_round} in {perf_src}"
        )
    labeled_size = int(cut_entry.get("labeled_size"))
    if labeled_size <= 0:
        raise SystemExit(f"Invalid labeled_size={labeled_size} at round={resume_round}")

    perf_upto = [r for r in perf if int(r.get("round", 0) or 0) <= resume_round]
    perf_upto.sort(key=lambda r: int(r.get("round", 0) or 0))
    if len(perf_upto) != resume_round:
        raise SystemExit(
            f"Expected perf length {resume_round} (round 1..{resume_round}), got {len(perf_upto)}"
        )

    source_pools_dir = pools_dir / run_id / source_exp
    labeled_df, unlabeled_df = _split_pools_at_labeled_size(source_pools_dir, labeled_size)

    for target_exp in target_exps:
        target_pools_dir = pools_dir / run_id / target_exp
        ckpt_path = checkpoints_dir / run_id / f"{target_exp}_state.json"

        if not args.force:
            if target_pools_dir.exists():
                raise SystemExit(f"Target pools already exist: {target_pools_dir}")
            if ckpt_path.exists():
                raise SystemExit(f"Target checkpoint already exists: {ckpt_path}")

        _write_pools(target_pools_dir, labeled_df, unlabeled_df)
        ckpt_payload = _build_checkpoint_payload(
            target_exp=target_exp,
            perf_upto=perf_upto,
            resume_round=resume_round,
            labeled_size=int(len(labeled_df)),
            unlabeled_size=int(len(unlabeled_df)),
        )
        _write_json_atomic(ckpt_path, ckpt_payload)

    launch_python = str(args.launch_python).strip() if args.launch_python else sys.executable
    launch_main = str(args.launch_main).strip() if args.launch_main else "src/main.py"
    launch_data_dir = str(args.launch_data_dir).strip() if args.launch_data_dir else None
    launch_seed = int(args.launch_seed) if args.launch_seed is not None else None
    launch_det = None
    if args.launch_deterministic is not None:
        launch_det = bool(int(args.launch_deterministic))

    out = {
        "run_id": run_id,
        "source_exp": source_exp,
        "target_exps": target_exps,
        "start_round": start_round,
        "resume_round": resume_round,
        "labeled_size": int(len(labeled_df)),
        "unlabeled_size": int(len(unlabeled_df)),
        "perf_source": perf_src,
        "source_pools_dir": str(source_pools_dir),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if bool(args.launch):
        _launch(
            python_bin=launch_python,
            main_path=launch_main,
            run_id=run_id,
            experiments=target_exps,
            start_mode=str(args.launch_start),
            data_dir=launch_data_dir,
            random_seed=launch_seed,
            deterministic=launch_det,
            max_concurrent=int(args.launch_max_concurrent),
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
