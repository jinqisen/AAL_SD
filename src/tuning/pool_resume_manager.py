import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class PoolResumeManager:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.pools_dir = self.results_dir / "pools"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.runs_dir = self.results_dir / "runs"

    def branch_from_round(
        self,
        source_run_id: str,
        source_exp: str,
        target_run_id: str,
        target_exps: List[str],
        branch_round: int,
    ) -> Dict:
        if branch_round < 2:
            raise ValueError(f"branch_round must be >= 2, got {branch_round}")
        perf, perf_src = self._resolve_source_perf(source_run_id, source_exp)
        resume_round = branch_round - 1
        cut_entry = self._get_perf_entry(perf, resume_round)
        if not cut_entry:
            raise RuntimeError(f"No perf entry for round={resume_round}")
        labeled_size = int(cut_entry["labeled_size"])
        if labeled_size <= 0:
            raise ValueError(
                f"Invalid labeled_size={labeled_size} at round={resume_round}"
            )
        perf_upto = [r for r in perf if int(r.get("round", 0)) <= resume_round]
        perf_upto.sort(key=lambda r: int(r.get("round", 0)))
        source_pools = self.pools_dir / source_run_id / source_exp
        labeled_df, unlabeled_df = self._split_pools_at_labeled_size(
            source_pools, labeled_size
        )
        results = []
        for target_exp in target_exps:
            target_pools = self.pools_dir / target_run_id / target_exp
            ckpt_path = (
                self.checkpoints_dir / target_run_id / f"{target_exp}_state.json"
            )
            self._write_pools(target_pools, labeled_df, unlabeled_df)
            manifest_src = source_pools / "pools_manifest.json"
            manifest_dst = target_pools / "pools_manifest.json"
            if manifest_src.exists() and not manifest_dst.exists():
                shutil.copy2(manifest_src, manifest_dst)
            ckpt_payload = self._build_checkpoint_payload(
                target_exp, perf_upto, resume_round, len(labeled_df), len(unlabeled_df)
            )
            self._write_json_atomic(ckpt_path, ckpt_payload)
            results.append(
                {
                    "target_exp": target_exp,
                    "pools_dir": str(target_pools),
                    "checkpoint": str(ckpt_path),
                    "labeled_size": len(labeled_df),
                    "resume_from_round": resume_round,
                }
            )
        return {
            "source": f"{source_run_id}/{source_exp}",
            "branch_round": branch_round,
            "targets": results,
        }

    def _resolve_source_perf(
        self, run_id: str, exp_name: str
    ) -> Tuple[List[Dict], str]:
        ckpt_path = self.checkpoints_dir / run_id / f"{exp_name}_state.json"
        if ckpt_path.exists():
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            perf = payload.get("performance_history", [])
            if perf:
                return perf, str(ckpt_path)
        result_path = self.runs_dir / run_id / f"result_{exp_name}.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            section = payload.get(exp_name, {})
            perf = section.get("performance_history", [])
            if perf:
                return perf, str(result_path)
        raise RuntimeError(
            f"Unable to find performance_history for {exp_name} in {run_id}"
        )

    def _get_perf_entry(self, perf: List[Dict], round_idx: int) -> Optional[Dict]:
        for r in perf:
            if int(r.get("round", 0)) == round_idx:
                return r
        return None

    def _split_pools_at_labeled_size(
        self, pools_dir: Path, labeled_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labeled_path = pools_dir / "labeled_pool.csv"
        unlabeled_path = pools_dir / "unlabeled_pool.csv"
        labeled_df = pd.read_csv(labeled_path)
        unlabeled_df = pd.read_csv(unlabeled_path)
        if labeled_size > len(labeled_df):
            raise ValueError(
                f"labeled_size={labeled_size} exceeds available labeled data "
                f"({len(labeled_df)} rows) in {pools_dir}"
            )
        labeled_head = labeled_df.iloc[:labeled_size].copy()
        labeled_tail = labeled_df.iloc[labeled_size:].copy()
        unlabeled_at_cut = pd.concat([unlabeled_df, labeled_tail], ignore_index=True)
        return labeled_head, unlabeled_at_cut

    def _write_pools(
        self, target_dir: Path, labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame
    ):
        target_dir.mkdir(parents=True, exist_ok=True)
        labeled_tmp = target_dir / "labeled_pool.csv.tmp"
        unlabeled_tmp = target_dir / "unlabeled_pool.csv.tmp"
        labeled_df.to_csv(labeled_tmp, index=False)
        unlabeled_df.to_csv(unlabeled_tmp, index=False)
        os.replace(labeled_tmp, target_dir / "labeled_pool.csv")
        os.replace(unlabeled_tmp, target_dir / "unlabeled_pool.csv")

    def _build_checkpoint_payload(
        self,
        exp_name: str,
        perf_upto: List[Dict],
        resume_round: int,
        labeled_size: int,
        unlabeled_size: int,
    ) -> Dict:
        budget_history = [int(r.get("labeled_size", 0)) for r in perf_upto]
        return {
            "round": resume_round,
            "performance_history": list(perf_upto),
            "budget_history": budget_history[:resume_round],
            "labeled_size": labeled_size,
            "unlabeled_size": unlabeled_size,
            "rng_states": {},
            "model_selection": "best_val",
            "experiment_name": exp_name,
        }

    def _write_json_atomic(self, path: Path, payload: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
