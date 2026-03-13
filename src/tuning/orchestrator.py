import os
import sys
import json
import copy
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tuning.analyzer import ExperimentAnalyzer
    from tuning.advisor import TuningAdvisor
    from tuning.proposer import (
        ExperimentProposer,
        BASE_CONFIG,
        VALID_THRESHOLD_PARAMS,
        VALID_POLICY_PARAMS,
    )
    from tuning.resource_monitor import ResourceMonitor
    from tuning.pool_resume_manager import PoolResumeManager
    from tuning.git_branch_manager import GitBranchManager
    from tuning.convergence import ConvergenceDetector
else:
    from .analyzer import ExperimentAnalyzer
    from .advisor import TuningAdvisor
    from .proposer import (
        ExperimentProposer,
        BASE_CONFIG,
        VALID_THRESHOLD_PARAMS,
        VALID_POLICY_PARAMS,
    )
    from .resource_monitor import ResourceMonitor
    from .pool_resume_manager import PoolResumeManager
    from .git_branch_manager import GitBranchManager
    from .convergence import ConvergenceDetector

# Params that only affect late-stage behavior — safe to branch from round 7
_LATE_STAGE_PARAMS = frozenset(
    {
        "late_stage_ramp",
        "selection_guardrail",
        "epochs_per_round_override",
        "LAMBDA_CLAMP_MAX",
        "LAMBDA_CLAMP_MIN",
    }
)

# Params that affect early-stage behavior — must run full
_EARLY_STAGE_PARAMS = frozenset(
    {
        "uncertainty_only_rounds",
        "warmup_start_round",
        "warmup_rounds",
        "warmup_lambda",
        "risk_control_start_round",
    }
)

# Params that affect risk control — branch from round 4
_RISK_CONTROL_PARAMS = frozenset(
    {
        "OVERFIT_RISK_HI",
        "OVERFIT_RISK_LO",
        "OVERFIT_TVC_MIN_HI",
        "LAMBDA_DELTA_UP",
        "LAMBDA_DELTA_DOWN",
        "LAMBDA_DOWN_COOLING_ROUNDS",
        "OVERFIT_RISK_EMA_ALPHA",
        "lambda_smoothing_alpha",
        "lambda_max_step",
    }
)

# Base ablation config template (mirrors full_model_A_lambda_policy_ab_tune_hi_ep10)
_BASE_ABLATION_TEMPLATE = {
    "use_agent": True,
    "sampler_type": "ad_kucs",
    "lambda_override": None,
    "acquisition_protocol": {
        "uncertainty_aggregation": "mean",
        "diversity_postprocess": "none",
    },
    "enable_l3_selection_logging": True,
    "enable_agent_prompt_logging": True,
    "l3_topk": 256,
    "l3_max_selected": 256,
    "rollback_config": {
        "mode": "adaptive_threshold",
        "std_factor": 1.5,
        "tau_min": 0.005,
    },
    "control_permissions": {
        "set_lambda": False,
        "set_query_size": False,
        "set_epochs_per_round": False,
        "set_alpha": False,
    },
}


# TODO(#23): implement analyze_selection_mask_distribution.py for offline mask distribution analysis
# TODO(#24): add sliding-window cross-iteration analysis in _collect_results (design Section 4.9)
# TODO(#25): integrate compute_budget() into _execute_experiments for Phase A/B budget management


class TuningOrchestrator:
    def __init__(self, config: Dict):
        self.target_miou = config.get("target_miou", 0.74)
        self.max_iterations = config.get("max_iterations", 10)
        self.seeds = config.get("seeds", [42, 43, 44])
        self.run_id_prefix = config.get("run_id_prefix", "autotune")
        self.results_dir = config.get("results_dir", "results")
        self.repo_dir = config.get("repo_dir", ".")
        self.max_concurrent = config.get("max_concurrent", 2)
        self.dry_run = bool(config.get("dry_run", False))
        self.analyzer = ExperimentAnalyzer(self.results_dir)
        llm_config = config.get("llm_advisor")
        self.advisor = (
            TuningAdvisor(llm_config, config.get("enable_llm_advisor", True))
            if llm_config
            else None
        )
        self.proposer = ExperimentProposer(self.advisor)
        self.monitor = ResourceMonitor(
            per_exp_gb=config.get("per_experiment_gb", 3.5),
            max_concurrent=self.max_concurrent,
        )
        self.pool_mgr = PoolResumeManager(self.results_dir)
        self.git_mgr = GitBranchManager(self.repo_dir)
        self.convergence = ConvergenceDetector(
            self.target_miou, max_iterations=self.max_iterations
        )
        self.iteration_history: List[Dict] = []

    def run(
        self, initial_run_id: Optional[str] = None, initial_exp: Optional[str] = None
    ):
        iteration = 0
        best_miou = 0.0
        best_config = None
        best_run_id = initial_run_id
        best_exp = initial_exp

        # Initialize diagnosis from the seed experiment (or empty if none provided)
        diagnosis: Dict = {"diagnostics": {"final_miou": best_miou}, "issues": []}
        if initial_run_id and initial_exp:
            # Validate initial experiment exists (#9)
            trace_path = os.path.join(
                self.results_dir, "runs", initial_run_id, f"{initial_exp}_trace.jsonl"
            )
            result_path = os.path.join(
                self.results_dir, "runs", initial_run_id, "experiment_results.json"
            )
            if not os.path.exists(trace_path) and not os.path.exists(result_path):
                raise FileNotFoundError(
                    f"Initial experiment not found: run_id={initial_run_id}, exp={initial_exp}. "
                    f"Checked: {trace_path} and {result_path}"
                )
            data = self.analyzer.load_experiment(initial_run_id, initial_exp)
            diag = self.analyzer.compute_diagnostics(data)
            if not diag:
                raise ValueError(
                    f"Initial experiment has no analyzable data: "
                    f"run_id={initial_run_id}, exp={initial_exp}"
                )
            diagnosis = self.analyzer.diagnose(diag)
            best_miou = diag.get("final_miou", 0.0)

        while iteration < self.max_iterations and best_miou < self.target_miou:
            print(
                f"\n=== Tuning Iteration {iteration} | Best mIoU: {best_miou:.4f} | Target: {self.target_miou} ==="
            )
            self.git_mgr.create_tuning_branch(iteration)
            proposals = self.proposer.propose(
                diagnosis,
                iteration,
                self.iteration_history,
            )
            self.git_mgr.commit_iteration_report(
                iteration,
                {**diagnosis, "best_miou": best_miou},
                proposals,
            )
            run_id = f"{self.run_id_prefix}_iter{iteration:03d}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self._execute_experiments(proposals, run_id, best_run_id, best_exp)
            iter_result = self._collect_results(run_id, proposals)

            if len(self.seeds) > 1 and iter_result.get("best_exp"):
                phase_b = self._run_phase_b_validation(
                    run_id, iter_result["best_exp"], iter_result["best_miou"]
                )
                if phase_b is not None:
                    iter_result["best_miou"] = phase_b["avg_miou"]
                    iter_result["phase_b"] = phase_b

            if iter_result["best_miou"] > best_miou:
                best_miou = iter_result["best_miou"]
                best_config = iter_result["best_config"]
                best_run_id = run_id
                best_exp = iter_result["best_exp"]

            if iter_result.get("diagnosis"):
                diagnosis = iter_result["diagnosis"]
            elif iter_result.get("best_exp") and iter_result["best_miou"] > 0:
                try:
                    _data = self.analyzer.load_experiment(
                        run_id, iter_result["best_exp"]
                    )
                    _diag = self.analyzer.compute_diagnostics(_data)
                    if _diag:
                        diagnosis = self.analyzer.diagnose(_diag)
                except Exception:
                    print(
                        "  [warn] Could not update diagnosis, reusing previous iteration's"
                    )
            else:
                print(
                    "  [warn] No successful experiments this iteration, reusing previous diagnosis"
                )

            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "best_miou": best_miou,
                    "config": best_config,
                    "direction": iter_result.get("best_direction", ""),
                    "run_id": run_id,
                }
            )

            conv_result = self.convergence.update(iteration, best_miou)
            if conv_result["action"] == "STOP":
                print(f"\nSTOP: {conv_result['reason']}")
                break
            if conv_result["action"] == "WARN":
                print(
                    f"WARN: {conv_result['reason']} (avg_delta={conv_result.get('avg_delta', 0):.4f})"
                )
            iteration += 1

        return {
            "iterations": iteration + 1,
            "best_miou": best_miou,
            "best_config": best_config,
            "target_reached": best_miou >= self.target_miou,
        }

    def _run_phase_b_validation(
        self, run_id: str, best_exp: str, phase_a_miou: float
    ) -> Optional[Dict]:
        """Phase B: validate best experiment with additional seeds."""
        phase_a_seed = self.seeds[0] if self.seeds else 43
        extra_seeds = [s for s in self.seeds if s != phase_a_seed]
        if not extra_seeds:
            return None
        print(f"  [phase_b] Validating {best_exp} with seeds {extra_seeds}")
        mious = [phase_a_miou]
        for seed in extra_seeds:
            seed_run_id = f"{run_id}_seed{seed}"
            self._run_batch([best_exp], seed_run_id, seed, start_mode="fresh")
            try:
                data = self.analyzer.load_experiment(seed_run_id, best_exp)
                diag = self.analyzer.compute_diagnostics(data)
                seed_miou = diag.get("final_miou", 0.0)
                if seed_miou > 0:
                    mious.append(seed_miou)
            except Exception as e:
                print(f"  [phase_b] Failed to collect seed {seed}: {e}")
        if len(mious) <= 1:
            return None
        import numpy as np

        return {
            "seeds_tested": [phase_a_seed] + extra_seeds,
            "mious": mious,
            "avg_miou": float(np.mean(mious)),
            "std_miou": float(np.std(mious)),
        }

    # ------------------------------------------------------------------
    # Execution layer
    # ------------------------------------------------------------------

    def _execute_experiments(
        self,
        proposals: List[Dict],
        run_id: str,
        source_run_id: Optional[str],
        source_exp: Optional[str],
    ):
        """Convert proposals to ablation configs, prepare pools, and run experiments."""
        if not proposals:
            return

        # Build ablation configs and classify each proposal
        ablation_configs: Dict[str, Dict] = {}
        branch_exps: List[Tuple[str, int]] = []  # (exp_name, branch_round)
        full_exps: List[str] = []

        for i, proposal in enumerate(proposals):
            exp_name = proposal.get(
                "experiment_name",
                f"auto_tune_iter{len(self.iteration_history):02d}_{proposal.get('direction', 'unknown')}_{i:02d}",
            )
            ablation_cfg = self._proposal_to_ablation_config(proposal)
            ablation_configs[exp_name] = ablation_cfg

            strategy = self._classify_run_strategy(proposal)
            if strategy["type"] == "branch":
                branch_exps.append((exp_name, strategy["branch_round"]))
            else:
                full_exps.append(exp_name)

        # Write sidecar configs so run_parallel_strict can pick them up
        self._write_sidecar_configs(ablation_configs, run_id)

        # Prepare pools for branch experiments
        if source_run_id and source_exp:
            for exp_name, branch_round in branch_exps:
                try:
                    self.pool_mgr.branch_from_round(
                        source_run_id=source_run_id,
                        source_exp=source_exp,
                        target_run_id=run_id,
                        target_exps=[exp_name],
                        branch_round=branch_round,
                    )
                    print(
                        f"  [pool] Branched {exp_name} from {source_run_id}/{source_exp} at round {branch_round}"
                    )
                except Exception as e:
                    print(
                        f"  [pool] Branch failed for {exp_name}: {e} — falling back to full run"
                    )
                    full_exps.append(exp_name)
        else:
            # No source to branch from — run everything fresh
            for exp_name, _ in branch_exps:
                full_exps.append(exp_name)
            branch_exps = []

        # Execute Phase A: single seed, all proposals
        phase_a_seed = self.seeds[0] if self.seeds else 43
        all_exp_names = [e for e, _ in branch_exps] + full_exps

        # Determine start mode per experiment
        branch_exp_names = {e for e, _ in branch_exps}
        resume_exps = [e for e in all_exp_names if e in branch_exp_names]
        fresh_exps = [e for e in all_exp_names if e not in branch_exp_names]

        if resume_exps:
            self._run_batch(resume_exps, run_id, phase_a_seed, start_mode="resume")
        if fresh_exps:
            self._run_batch(fresh_exps, run_id, phase_a_seed, start_mode="fresh")

    def _run_batch(
        self,
        exp_names: List[str],
        run_id: str,
        seed: int,
        start_mode: str = "fresh",
    ):
        """Run a batch of experiments via run_parallel_strict.py subprocess."""
        if not exp_names:
            return

        import time as _time

        for _wait in range(30):
            if not self.monitor.should_pause():
                break
            print(f"  [exec] Memory low, waiting 10s before launch... ({_wait + 1}/30)")
            _time.sleep(10)
        slots = self.monitor.get_available_slots(0)
        workers = max(1, min(slots, len(exp_names)))

        src_dir = Path(self.repo_dir) / "src"
        script = str(src_dir / "run_parallel_strict.py")
        include_str = ",".join(exp_names)

        cmd = [
            sys.executable,
            script,
            "--run-id",
            run_id,
            "--include",
            include_str,
            "--seed",
            str(seed),
            "--execution",
            "parallel" if workers > 1 else "sequential",
            "--agent-workers",
            str(workers),
        ]
        if start_mode == "resume":
            cmd += ["--resume", run_id]
        if self.dry_run:
            cmd += ["--dry-run"]

        print(
            f"  [exec] Running {len(exp_names)} experiments (workers={workers}, mode={start_mode}): {exp_names}"
        )
        try:
            subprocess.run(cmd, cwd=self.repo_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"  [exec] Batch run failed (exit {e.returncode}), continuing with result collection"
            )

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(self, run_id: str, proposals: List[Dict]) -> Dict:
        """Load and diagnose all experiments from this iteration, return best."""
        # Build exp_name -> proposal mapping
        exp_map: Dict[str, Dict] = {}
        for i, p in enumerate(proposals):
            exp_name = p.get(
                "experiment_name",
                f"auto_tune_iter{len(self.iteration_history):02d}_{p.get('direction', 'unknown')}_{i:02d}",
            )
            exp_map[exp_name] = p

        best_miou = 0.0
        best_exp_name: Optional[str] = None
        best_config: Optional[Dict] = None
        best_direction: str = ""
        best_diagnosis: Optional[Dict] = None

        for exp_name, proposal in exp_map.items():
            try:
                data = self.analyzer.load_experiment(run_id, exp_name)
                diag = self.analyzer.compute_diagnostics(data)
                miou = diag.get("final_miou", 0.0)
                if miou > best_miou:
                    best_miou = miou
                    best_exp_name = exp_name
                    best_config = proposal
                    best_direction = proposal.get("direction", "")
                    best_diagnosis = self.analyzer.diagnose(diag)
            except Exception as e:
                print(f"  [collect] Failed to analyze {exp_name}: {e}")

        # Fallback: try loading from experiment_results.json if trace analysis failed
        if best_miou == 0.0:
            quick_results = self.analyzer.load_all_experiments(run_id)
            if quick_results:
                best_exp_name, best_miou = max(
                    quick_results.items(), key=lambda x: x[1]
                )
                best_config = exp_map.get(best_exp_name, {})
                best_direction = best_config.get("direction", "")

        return {
            "best_exp": best_exp_name,
            "best_miou": best_miou,
            "best_config": best_config or {},
            "best_direction": best_direction,
            "diagnosis": best_diagnosis,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _proposal_to_ablation_config(self, proposal: Dict) -> Dict:
        """Convert a Proposer proposal dict to an ABLATION_SETTINGS-compatible config."""
        cfg = copy.deepcopy(_BASE_ABLATION_TEMPLATE)

        # Start from BASE_CONFIG thresholds and policy
        threshold_overrides = copy.deepcopy(
            BASE_CONFIG.get("agent_threshold_overrides", {})
        )
        lambda_policy = copy.deepcopy(BASE_CONFIG.get("lambda_policy", {}))

        # Apply threshold overrides from proposal
        for key, val in proposal.items():
            if key in VALID_THRESHOLD_PARAMS:
                threshold_overrides[key] = val
            elif key in VALID_POLICY_PARAMS:
                lambda_policy[key] = val

        # Apply structural params
        if "late_stage_ramp" in proposal:
            lambda_policy["late_stage_ramp"] = proposal["late_stage_ramp"]
        if "selection_guardrail" in proposal:
            lambda_policy["selection_guardrail"] = proposal["selection_guardrail"]
        if "epochs_per_round_override" in proposal:
            cfg["epochs_per_round_override"] = int(
                proposal["epochs_per_round_override"]
            )

        cfg["agent_threshold_overrides"] = threshold_overrides
        cfg["lambda_policy"] = lambda_policy
        cfg["description"] = (
            f"AutoTune: {proposal.get('direction', 'unknown')} — "
            + proposal.get("llm_metadata", {}).get("description", "rule-based proposal")
        )
        return cfg

    def _classify_run_strategy(self, proposal: Dict) -> Dict:
        """Decide branch vs full run based on which params the proposal changes."""
        changed = set(proposal.keys()) - {
            "direction",
            "experiment_name",
            "llm_metadata",
            "resume_strategy",
        }

        if "resume_strategy" in proposal:
            rs = proposal["resume_strategy"]
            if rs.get("type") == "branch":
                return {"type": "branch", "branch_round": rs.get("branch_round", 7)}

        if not changed:
            return {"type": "full"}

        has_early = bool(changed & _EARLY_STAGE_PARAMS)
        has_late = bool(changed & _LATE_STAGE_PARAMS)
        has_risk = bool(changed & _RISK_CONTROL_PARAMS)

        if has_early:
            return {"type": "full"}

        if has_late and has_risk:
            return {"type": "full"}

        if changed.issubset(_LATE_STAGE_PARAMS):
            return {"type": "branch", "branch_round": 7}

        if changed.issubset(_RISK_CONTROL_PARAMS):
            return {"type": "branch", "branch_round": 4}

        return {"type": "full"}

    def _write_sidecar_configs(self, configs: Dict[str, Dict], run_id: str):
        """Write auto-tuned experiment configs to sidecar JSON for the experiment runner."""
        sidecar_path = (
            Path(self.repo_dir) / "src" / "experiments" / "auto_tune_configs.json"
        )
        tmp = sidecar_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, sidecar_path)
        print(f"  [sidecar] Wrote {len(configs)} configs to {sidecar_path}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AAL-SD Auto-Tuning Orchestrator")
    parser.add_argument(
        "--initial-run-id", required=True, help="Source run_id to start from"
    )
    parser.add_argument(
        "--initial-exp", required=True, help="Source experiment name to start from"
    )
    parser.add_argument(
        "--target-miou", type=float, default=0.74, help="Target mIoU to reach"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Max tuning iterations"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[43], help="Seeds for Phase A screening"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2, help="Max concurrent experiments"
    )
    parser.add_argument(
        "--llm-config", type=str, default=None, help="Path to tuning LLM config JSON"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM advisor (pure rule mode)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--repo-dir", type=str, default=".", help="Repository root directory"
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default="autotune",
        help="Prefix for generated run IDs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate configs and manifests (no training execution)",
    )
    args = parser.parse_args()

    llm_cfg = None
    if not args.no_llm:
        cfg_path = args.llm_config or os.path.join(
            args.repo_dir, "src", "tuning_llm_config.json"
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                llm_cfg = json.load(f)
        else:
            print(
                f"[warn] LLM config not found at {cfg_path}, running in rule-only mode"
            )

    orchestrator_config = {
        "target_miou": args.target_miou,
        "max_iterations": args.max_iterations,
        "seeds": args.seeds,
        "run_id_prefix": args.run_id_prefix,
        "results_dir": args.results_dir,
        "repo_dir": args.repo_dir,
        "max_concurrent": args.max_concurrent,
        "llm_advisor": llm_cfg,
        "enable_llm_advisor": not args.no_llm,
        "dry_run": bool(args.dry_run),
    }

    orchestrator = TuningOrchestrator(orchestrator_config)
    result = orchestrator.run(
        initial_run_id=args.initial_run_id,
        initial_exp=args.initial_exp,
    )
    print(f"\n=== Tuning Complete ===")
    print(f"Iterations: {result['iterations']}")
    print(f"Best mIoU:  {result['best_miou']:.4f}")
    print(f"Target reached: {result['target_reached']}")
