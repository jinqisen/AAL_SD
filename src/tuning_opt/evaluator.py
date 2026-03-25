from __future__ import annotations

import math
import json
import re
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import read_json_dict


_RE_FINAL_TEST = re.compile(r"最终报告 mIoU\(test\):\s*([0-9.]+)")
_RE_FINAL_OUT = re.compile(r"最终输出 mIoU:\s*([0-9.]+)")
_RE_LAST_VAL = re.compile(r"最后一轮选模 mIoU\(val\):\s*([0-9.]+)")


@dataclass(frozen=True)
class EvalResult:
    exp_name: str
    miou: Optional[float]
    md_path: str


@dataclass
class MultiSeedResult:
    """Aggregated result across multiple seeds for F3 statistical review."""

    exp_name: str
    seed_mious: List[float]  # one entry per seed that completed

    @property
    def mean(self) -> Optional[float]:
        return sum(self.seed_mious) / len(self.seed_mious) if self.seed_mious else None

    @property
    def std(self) -> Optional[float]:
        if len(self.seed_mious) < 2:
            return None
        m = self.mean
        if m is None:
            return None
        variance = sum((x - m) ** 2 for x in self.seed_mious) / (
            len(self.seed_mious) - 1
        )
        return math.sqrt(variance)

    def ci_lower(self, z: float = 1.96) -> Optional[float]:
        """Lower bound of z-score confidence interval."""
        if self.mean is None or self.std is None:
            return self.mean
        return self.mean - z * self.std / math.sqrt(len(self.seed_mious))

    def ci_upper(self, z: float = 1.96) -> Optional[float]:
        if self.mean is None or self.std is None:
            return self.mean
        return self.mean + z * self.std / math.sqrt(len(self.seed_mious))

    @property
    def min_miou(self) -> Optional[float]:
        if not self.seed_mious:
            return None
        return min(self.seed_mious)


def parse_final_miou_from_md(md_path: Path) -> Optional[float]:
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    matches_test = _RE_FINAL_TEST.findall(text)
    if matches_test:
        try:
            return float(matches_test[-1])
        except Exception:
            pass
    matches_out = _RE_FINAL_OUT.findall(text)
    if matches_out:
        try:
            return float(matches_out[-1])
        except Exception:
            pass
    return None


def parse_last_val_miou_from_md(md_path: Path) -> Optional[float]:
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    matches = _RE_LAST_VAL.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def parse_objective_miou_from_md(md_path: Path, objective: str) -> Optional[float]:
    obj = str(objective or "").strip().lower()
    if obj in ("val", "best_val", "last_val"):
        v = parse_last_val_miou_from_md(md_path)
        if v is not None:
            return v
        return None
    return parse_final_miou_from_md(md_path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_dict(path)


def parse_objective_from_status(status_path: Path, objective: str) -> Optional[float]:
    status = _read_json(status_path)
    if not isinstance(status, dict):
        return None
    result = status.get("result")
    if not isinstance(result, dict):
        return None

    obj = str(objective or "").strip().lower()
    if obj in ("alc", "learning_curve", "learning_curve_area"):
        v = result.get("alc")
        return float(v) if isinstance(v, (int, float)) else None

    if obj in ("val", "best_val", "last_val", "final_val"):
        v = result.get("final_selected_mIoU")
        if isinstance(v, (int, float)):
            return float(v)
        v = result.get("final_mIoU")
        return float(v) if isinstance(v, (int, float)) else None

    if obj in ("test", "final_test", "report", "final_report"):
        v = result.get("final_report_mIoU")
        if isinstance(v, (int, float)):
            return float(v)
        v = result.get("final_mIoU")
        return float(v) if isinstance(v, (int, float)) else None

    if obj in ("miou", "final_miou"):
        v = result.get("final_mIoU")
        return float(v) if isinstance(v, (int, float)) else None

    if obj in ("f1", "final_f1"):
        v = result.get("final_f1")
        return float(v) if isinstance(v, (int, float)) else None

    return None


class ExperimentEvaluator:
    def __init__(self, *, repo_root: Path):
        self.repo_root = repo_root

    def run_batch(
        self,
        *,
        run_id: str,
        exp_names: List[str],
        seed: int,
        max_concurrent: int,
        n_rounds: int,
        epochs_per_round: Optional[int] = None,
        resume: bool = True,
        dynamic_agent_workers: bool = False,
        agent_workers_min: int = 1,
        reserve_free_gb: float = 6.0,
        mem_per_agent_worker_gb: float = 4.0,
        dynamic_poll_seconds: float = 2.0,
        timeout: Optional[int] = None,
    ) -> None:
        include = ",".join(exp_names)
        cmd = [
            "python",
            "src/run_parallel_strict.py",
            "--seed",
            str(int(seed)),
            "--include",
            include,
            "--agent-workers",
            str(int(max_concurrent)),
            "--non-agent-workers",
            "0",
            "--n-rounds",
            str(int(n_rounds)),
            "--owner-pid",
            str(int(os.getpid())),
        ]
        if bool(dynamic_agent_workers):
            cmd.extend(
                [
                    "--dynamic-agent-workers",
                    "--agent-workers-min",
                    str(int(agent_workers_min)),
                    "--reserve-free-gb",
                    str(float(reserve_free_gb)),
                    "--mem-per-agent-worker-gb",
                    str(float(mem_per_agent_worker_gb)),
                    "--dynamic-poll-seconds",
                    str(float(dynamic_poll_seconds)),
                ]
            )
        if epochs_per_round is not None:
            cmd.extend(["--epochs-per-round", str(int(epochs_per_round))])
        if resume:
            cmd.extend(["--resume", str(run_id)])
        else:
            cmd.extend(["--run-id", str(run_id)])
        p = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(int(p.pid), signal.SIGTERM)
            except Exception:
                pass
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if p.poll() is not None:
                    break
                time.sleep(0.05)
            if p.poll() is None:
                try:
                    os.killpg(int(p.pid), signal.SIGKILL)
                except Exception:
                    pass
            rc = int(p.poll() or 1)
        except KeyboardInterrupt:
            try:
                os.killpg(int(p.pid), signal.SIGTERM)
            except Exception:
                pass
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if p.poll() is not None:
                    break
                time.sleep(0.05)
            if p.poll() is None:
                try:
                    os.killpg(int(p.pid), signal.SIGKILL)
                except Exception:
                    pass
            raise

        if int(rc) != 0:
            import sys

            print(
                f"[evaluator] run_batch returned non-zero exit code {int(rc)} "
                f"for run_id={run_id} exp_names={exp_names}. "
                f"Partial results may be available via collect_results().",
                file=sys.stderr,
            )

    def extract_guardrail_stats(self, run_id: str, exp_name: str) -> Dict[str, Any]:
        """Extract guardrail application statistics from trace files."""
        run_dir = self.repo_root / "results" / "runs" / str(run_id)
        trace_path = run_dir / f"{exp_name}_trace.jsonl"
        stats = {"applied": 0, "total": 0, "fraction": None}
        if not trace_path.exists():
            return stats
        
        applied_count = 0
        rounds_seen = set()
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        r = data.get("round", 0)
                        if r:
                            rounds_seen.add(r)
                        if data.get("type") == "selection_guardrail" and data.get("applied") is True:
                            applied_count += 1
                    except Exception:
                        pass
            total_rounds = max(rounds_seen) if rounds_seen else 0
            stats["applied"] = applied_count
            stats["total"] = total_rounds
            if total_rounds > 0:
                stats["fraction"] = applied_count / total_rounds
        except Exception:
            pass
        return stats

    def run_multi_seed(
        self,
        *,
        run_id: str,
        exp_names: List[str],
        seeds: List[int],
        max_concurrent: int,
        n_rounds: int,
        epochs_per_round: Optional[int] = None,
        resume: bool = True,
    ) -> None:
        for seed in seeds:
            self.run_batch(
                run_id=run_id,
                exp_names=exp_names,
                seed=seed,
                max_concurrent=max_concurrent,
                n_rounds=n_rounds,
                epochs_per_round=epochs_per_round,
                resume=resume,
            )

    def collect_results(
        self, *, run_id: str, exp_names: Iterable[str], objective: str = "test"
    ) -> Dict[str, EvalResult]:
        out: Dict[str, EvalResult] = {}
        run_dir = self.repo_root / "results" / "runs" / run_id
        for exp_name in exp_names:
            status_path = run_dir / f"{exp_name}_status.json"
            md_path = run_dir / f"{exp_name}.md"
            miou = parse_objective_from_status(status_path, objective)
            if miou is None:
                miou = parse_objective_miou_from_md(md_path, objective)
            out[exp_name] = EvalResult(
                exp_name=str(exp_name), miou=miou, md_path=str(md_path)
            )
        return out

    def collect_multi_seed_results(
        self,
        *,
        run_ids_by_seed: Dict[int, str],
        exp_names: Iterable[str],
        objective: str = "test",
    ) -> Dict[str, MultiSeedResult]:
        out: Dict[str, MultiSeedResult] = {}
        for exp_name in exp_names:
            seed_mious: List[float] = []
            for seed, run_id in sorted(run_ids_by_seed.items(), key=lambda x: x[0]):
                run_dir = self.repo_root / "results" / "runs" / str(run_id)
                status_path = run_dir / f"{exp_name}_status.json"
                md_path = run_dir / f"{exp_name}.md"
                miou = parse_objective_from_status(status_path, objective)
                if miou is None:
                    miou = parse_objective_miou_from_md(md_path, objective)
                if miou is not None:
                    seed_mious.append(miou)
            out[exp_name] = MultiSeedResult(
                exp_name=str(exp_name), seed_mious=seed_mious
            )
        return out
