from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_RE_FINAL_TEST = re.compile(r"最终报告 mIoU\(test\):\s*([0-9.]+)")
_RE_FINAL_OUT = re.compile(r"最终输出 mIoU:\s*([0-9.]+)")


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


def parse_final_miou_from_md(md_path: Path) -> Optional[float]:
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = _RE_FINAL_TEST.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = _RE_FINAL_OUT.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
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
        ]
        if epochs_per_round is not None:
            cmd.extend(["--epochs-per-round", str(int(epochs_per_round))])
        if resume:
            cmd.extend(["--resume", str(run_id)])
        else:
            cmd.extend(["--run-id", str(run_id)])
        subprocess.run(cmd, cwd=str(self.repo_root), check=True)

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
        self, *, run_id: str, exp_names: Iterable[str]
    ) -> Dict[str, EvalResult]:
        out: Dict[str, EvalResult] = {}
        run_dir = self.repo_root / "results" / "runs" / run_id
        for exp_name in exp_names:
            md_path = run_dir / f"{exp_name}.md"
            miou = parse_final_miou_from_md(md_path)
            out[exp_name] = EvalResult(
                exp_name=str(exp_name), miou=miou, md_path=str(md_path)
            )
        return out

    def collect_multi_seed_results(
        self, *, run_id: str, exp_names: Iterable[str], seeds: List[int]
    ) -> Dict[str, MultiSeedResult]:
        out: Dict[str, MultiSeedResult] = {}
        run_dir = self.repo_root / "results" / "runs" / run_id
        for exp_name in exp_names:
            seed_mious: List[float] = []
            for seed in seeds:
                md_path = run_dir / f"{exp_name}.md"
                miou = parse_final_miou_from_md(md_path)
                if miou is not None:
                    seed_mious.append(miou)
            out[exp_name] = MultiSeedResult(
                exp_name=str(exp_name), seed_mious=seed_mious
            )
        return out
