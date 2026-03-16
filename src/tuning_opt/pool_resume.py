from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_write_json, read_json_dict


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_dict(path)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, indent=2)


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


@dataclass(frozen=True)
class BranchSpec:
    branch_round: int
    resume_round: int
    labeled_size: int
    unlabeled_size: int


class PoolResumeManager:
    def __init__(self, *, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.pools_dir = self.results_dir / "pools"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.runs_dir = self.results_dir / "runs"

    def copy_base_pools(self, source_run_id: str, target_run_id: str) -> Optional[str]:
        src_dir = self.pools_dir / source_run_id / "_base"
        if not src_dir.exists():
            return None
        dst_dir = self.pools_dir / target_run_id / "_base"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ("labeled_pool.csv", "unlabeled_pool.csv", "pools_manifest.json"):
            src = src_dir / name
            dst = dst_dir / name
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())
        return str(dst_dir)

    def branch_from_round(
        self,
        *,
        source_run_id: str,
        source_exp: str,
        target_run_id: str,
        target_exps: List[str],
        branch_round: int,
    ) -> Dict[str, Any]:
        if int(branch_round) < 2:
            raise ValueError(f"branch_round must be >= 2, got {branch_round}")

        self.copy_base_pools(source_run_id, target_run_id)

        perf, perf_src = self._resolve_source_perf(source_run_id, source_exp)
        max_round = max((int(r.get("round", 0) or 0) for r in perf), default=0)
        desired_resume_round = int(branch_round) - 1
        resume_round = min(desired_resume_round, max_round)
        cut_entry = self._get_perf_entry(perf, resume_round)
        if not cut_entry:
            raise RuntimeError(
                f"No perf entry for round={resume_round} (max_round={max_round}, desired_resume_round={desired_resume_round})"
            )
        labeled_size = int(cut_entry.get("labeled_size", 0) or 0)
        if labeled_size <= 0:
            raise ValueError(f"Invalid labeled_size={labeled_size} at round={resume_round}")

        perf_upto = [r for r in perf if int(r.get("round", 0) or 0) <= resume_round]
        perf_upto.sort(key=lambda r: int(r.get("round", 0) or 0))

        source_pools = self.pools_dir / source_run_id / source_exp
        labeled_rows, unlabeled_rows, fields = self._split_pools_at_labeled_size(
            source_pools, labeled_size
        )

        results = []
        for target_exp in target_exps:
            target_pools = self.pools_dir / target_run_id / target_exp
            target_pools.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(target_pools / "labeled_pool.csv", labeled_rows, fields)
            _write_csv_rows(target_pools / "unlabeled_pool.csv", unlabeled_rows, fields)

            manifest_src = source_pools / "pools_manifest.json"
            manifest_dst = target_pools / "pools_manifest.json"
            if manifest_src.exists() and not manifest_dst.exists():
                manifest_dst.write_bytes(manifest_src.read_bytes())

            ckpt_path = self.checkpoints_dir / target_run_id / f"{target_exp}_state.json"
            ckpt_payload = self._build_checkpoint_payload(
                exp_name=target_exp,
                perf_upto=perf_upto,
                resume_round=resume_round,
                labeled_size=len(labeled_rows),
                unlabeled_size=len(unlabeled_rows),
            )
            _write_json_atomic(ckpt_path, ckpt_payload)
            results.append(
                {
                    "target_exp": target_exp,
                    "pools_dir": str(target_pools),
                    "checkpoint": str(ckpt_path),
                    "resume_from_round": resume_round,
                }
            )

        return {
            "source": f"{source_run_id}/{source_exp}",
            "perf_source": perf_src,
            "branch_round": int(branch_round),
            "resume_round": int(resume_round),
            "labeled_size": int(len(labeled_rows)),
            "unlabeled_size": int(len(unlabeled_rows)),
            "targets": results,
        }

    def _resolve_source_perf(self, run_id: str, exp_name: str) -> Tuple[List[Dict[str, Any]], str]:
        ckpt_path = self.checkpoints_dir / run_id / f"{exp_name}_state.json"
        ckpt = _read_json(ckpt_path)
        if isinstance(ckpt, dict):
            perf = ckpt.get("performance_history")
            if isinstance(perf, list) and perf:
                return perf, str(ckpt_path)

        result_path = self.runs_dir / run_id / f"result_{exp_name}.json"
        res = _read_json(result_path)
        if isinstance(res, dict):
            section = res.get(exp_name)
            if isinstance(section, dict):
                perf = section.get("performance_history")
                if isinstance(perf, list) and perf:
                    return perf, str(result_path)

        table_path = self.runs_dir / run_id / "experiment_results.json"
        table = _read_json(table_path)
        if isinstance(table, dict):
            section = table.get(exp_name)
            if isinstance(section, dict):
                perf = section.get("performance_history")
                if isinstance(perf, list) and perf:
                    return perf, str(table_path)

        raise RuntimeError(f"Unable to find performance_history for {exp_name} in {run_id}")

    @staticmethod
    def _get_perf_entry(perf: List[Dict[str, Any]], round_idx: int) -> Optional[Dict[str, Any]]:
        for r in perf:
            if int(r.get("round", 0) or 0) == int(round_idx):
                return r
        return None

    def _split_pools_at_labeled_size(
        self, pools_dir: Path, labeled_size: int
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        labeled_path = pools_dir / "labeled_pool.csv"
        unlabeled_path = pools_dir / "unlabeled_pool.csv"
        labeled_rows, fields = _read_csv_rows(labeled_path)
        unlabeled_rows, fields2 = _read_csv_rows(unlabeled_path)
        if fields2 and not fields:
            fields = fields2
        if labeled_size > len(labeled_rows):
            raise ValueError(
                f"labeled_size={labeled_size} exceeds available labeled data "
                f"({len(labeled_rows)} rows) in {pools_dir}"
            )
        labeled_head = labeled_rows[:labeled_size]
        labeled_tail = labeled_rows[labeled_size:]
        unlabeled_at_cut = list(unlabeled_rows) + list(labeled_tail)
        return labeled_head, unlabeled_at_cut, fields

    @staticmethod
    def _build_checkpoint_payload(
        *,
        exp_name: str,
        perf_upto: List[Dict[str, Any]],
        resume_round: int,
        labeled_size: int,
        unlabeled_size: int,
    ) -> Dict[str, Any]:
        budget_history: List[int] = []
        for r in perf_upto:
            try:
                budget_history.append(int(r.get("labeled_size", 0) or 0))
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
            "experiment_name": str(exp_name),
        }
