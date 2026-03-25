import os
import sys
import json
import shutil
import glob
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from main import ActiveLearningPipeline
from experiments.ablation_config import ABLATION_SETTINGS

def branch_and_run(run_id, branch_from_run_id, round_to_branch, lambdas):
    results_dir = Path("results")
    base_run_dir = results_dir / "runs" / branch_from_run_id
    new_run_dir = results_dir / "runs" / run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Check base run
    if not base_run_dir.exists():
        print(f"Base run {branch_from_run_id} not found!")
        return

    # Branch checkpoints
    base_ckpt_dir = results_dir / "checkpoints" / branch_from_run_id / "fixed_lambda_round_models"
    if not base_ckpt_dir.exists():
        # Fallback to runs dir if that's where it is
        base_ckpt_dir = base_run_dir / "fixed_lambda_round_models"
    
    for lam in lambdas:
        lam_str = f"{lam:.1f}".replace(".", "")
        exp_name = f"test_k_prime_lam_{lam_str}"
        
        # We need to create a config for this experiment
        exp_config = dict(ABLATION_SETTINGS["fixed_lambda"])
        exp_config["lambda_override"] = lam
        ABLATION_SETTINGS[exp_name] = exp_config
        
        # Setup directories for the new experiment
        exp_ckpt_dir = new_run_dir / f"{exp_name}_round_models"
        exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint up to branch_round
        for r in range(1, round_to_branch + 1):
            src_ckpt = base_ckpt_dir / f"round_{r:02d}_best_val.pt"
            dst_ckpt = exp_ckpt_dir / f"round_{r:02d}_best_val.pt"
            if src_ckpt.exists():
                shutil.copy2(src_ckpt, dst_ckpt)
                
        # Copy pipeline state
        src_state = results_dir / "checkpoints" / branch_from_run_id / "fixed_lambda_state.json"
        dst_state_dir = results_dir / "checkpoints" / run_id
        dst_state_dir.mkdir(parents=True, exist_ok=True)
        dst_state = dst_state_dir / f"{exp_name}_state.json"
        if src_state.exists():
            with open(src_state, "r") as f:
                state_data = json.load(f)
            # Truncate state history up to round_to_branch
            if "performance_history" in state_data:
                state_data["performance_history"] = [x for x in state_data["performance_history"] if x["round"] <= round_to_branch]
            state_data["round"] = round_to_branch
            with open(dst_state, "w") as f:
                json.dump(state_data, f)
                
        # We also need to copy the pools state.
        # Pools are in results/pools/{branch_from_run_id}/fixed_lambda
        src_pool_dir = results_dir / "pools" / branch_from_run_id / "fixed_lambda"
        dst_pool_dir = results_dir / "pools" / run_id / exp_name
        if dst_pool_dir.exists():
            shutil.rmtree(dst_pool_dir)
        dst_pool_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_pool_dir, dst_pool_dir)
        
        # Copy log file to trick it into resuming
        src_log = results_dir / "logs_md" / f"fixed_lambda_{branch_from_run_id}.md"
        dst_log = results_dir / "logs_md" / f"{exp_name}_{run_id}.md"
        if src_log.exists():
            # Only keep up to round 5
            with open(src_log, "r") as f:
                content = f.read()
            # Split by ## Round
            parts = content.split("## Round ")
            new_content = parts[0]
            for part in parts[1:]:
                r_num = int(part.split("\n")[0].strip())
                if r_num <= round_to_branch:
                    new_content += "## Round " + part
            with open(dst_log, "w") as f:
                f.write(new_content)
                
        print(f"--- Running {exp_name} with lambda={lam} ---")
        cfg = Config()
        cfg.N_ROUNDS = round_to_branch + 2
        cfg.RESULTS_DIR = str(results_dir)
        cfg.POOLS_DIR = str(results_dir / "pools")
        cfg.CHECKPOINT_DIR = str(results_dir / "checkpoints")
        cfg.START_MODE = "resume"
        
        pipeline = ActiveLearningPipeline(cfg, exp_name, run_id=run_id)
        pipeline.run()

if __name__ == "__main__":
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    branch_and_run("k_prime_test_01", "lambda_sweep_experiment_01", 5, lambdas)
