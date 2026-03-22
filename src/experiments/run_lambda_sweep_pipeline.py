import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="End-to-End Lambda Sweep Pipeline")
    parser.add_argument("--run-id", required=True, help="Run ID for this sweep experiment")
    parser.add_argument("--trunk-exp", default="fixed_lambda", help="Base experiment to use as trunk (e.g. fixed_lambda)")
    parser.add_argument("--trunk-rounds", type=int, default=10, help="How many rounds to run the trunk")
    parser.add_argument("--sweep-rounds", type=str, default="3,6,9", help="Comma-separated rounds to perform sweep ON")
    parser.add_argument("--lambdas", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0", help="Comma-separated lambdas")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers for branches")
    args = parser.parse_args()

    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    sweep_rounds = [int(x) for x in args.sweep_rounds.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print(f"=== Phase 1: Running Trunk Experiment ({args.trunk_exp}) ===")
    print("Setting AAL_SD_ROUND_MODEL_RETENTION=all to save all intermediate checkpoints.")
    
    env = os.environ.copy()
    env["AAL_SD_ROUND_MODEL_RETENTION"] = "all"
    
    trunk_cmd = [
        sys.executable,
        str(base_dir / "src/main.py"),
        "--run_id", args.run_id,
        "--experiment_name", args.trunk_exp,
        "--start", "fresh",
        "--n_rounds", str(args.trunk_rounds)
    ]
    
    checkpoints_dir = base_dir / "results/checkpoints" / args.run_id
    trunk_state_path = checkpoints_dir / f"{args.trunk_exp}_state.json"
    trunk_round_models_dir = base_dir / "results/runs" / args.run_id / f"{args.trunk_exp}_round_models"

    trunk_ready = False
    if trunk_state_path.exists() and trunk_round_models_dir.is_dir():
        try:
            trunk_state = json.loads(trunk_state_path.read_text(encoding="utf-8"))
            trunk_round = int(trunk_state.get("round", 0) or 0)
            if trunk_round >= int(args.trunk_rounds):
                need = [int(r) for r in range(1, int(args.trunk_rounds) + 1)]
                have = [
                    (trunk_round_models_dir / f"round_{int(r):02d}_best_val.pt").exists()
                    for r in need
                ]
                trunk_ready = bool(all(have))
        except Exception:
            trunk_ready = False

    if trunk_ready:
        print(f"Trunk already available: {trunk_state_path}")
    else:
        print(f"Running: {' '.join(trunk_cmd)}")
        subprocess.run(trunk_cmd, env=env, check=True)
    
    # 2. Update ablation_config.py for lambdas
    print("\n=== Phase 2: Injecting Sweep Configs ===")
    config_path = base_dir / "src/experiments/ablation_config.py"
    config_content = config_path.read_text()
    
    new_configs = []
    for R in sweep_rounds:
        for lam in lambdas:
            lam_str = str(lam).replace(".", "")
            exp_name = f"sweep_r{R}_lam_{lam_str}"
            if f'"{exp_name}":' not in config_content:
                new_configs.append(f"""
    "{exp_name}": {{
        "description": "Lambda sweep at Round {R}, lambda={lam}",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": {lam},
    }},""")
            
    if new_configs:
        # Find ABLATION_SETTINGS end
        # Find where `build_spec_from_legacy_dict` is defined. It's likely below ABLATION_SETTINGS.
        # We should insert before the closing brace of ABLATION_SETTINGS
        idx = config_content.find("ABLATION_SETTINGS = {")
        if idx != -1:
            # find the matching closing brace
            brace_count = 0
            for i in range(idx, len(config_content)):
                if config_content[i] == '{':
                    brace_count += 1
                elif config_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the end of ABLATION_SETTINGS
                        config_content = config_content[:i] + "".join(new_configs) + config_content[i:]
                        config_path.write_text(config_content)
                        print(f"Added {len(new_configs)} new configurations.")
                        break
        else:
            # Fallback if ABLATION_SETTINGS not found
            last_brace_idx = config_content.rfind('}')
            if last_brace_idx != -1:
                config_content = config_content[:last_brace_idx] + "".join(new_configs) + "\n}"
                config_path.write_text(config_content)
                print(f"Added {len(new_configs)} new configurations.")
            
    # VERY IMPORTANT: To fix the ImportError, ensure the python interpreter finds the updated code
    # We don't need to do anything special here as long as it's written properly.
    
    print("\n=== Phase 3: Branching and Running Sweeps ===")
    branch_script = base_dir / "src/utils/branch_experiment_from_round.py"
    main_script = base_dir / "src/main.py"
    
    procs = []
    
    for R in sweep_rounds:
        resume_round = R - 2
        if resume_round < 0:
            print(f"Skipping round {R}, must be >= 2")
            continue
            
        trunk_ckpt_path = (
            base_dir
            / "results/runs"
            / args.run_id
            / f"{args.trunk_exp}_round_models"
            / f"round_{R-1:02d}_best_val.pt"
        )
        if not trunk_ckpt_path.exists():
            raise FileNotFoundError(f"Missing trunk checkpoint: {trunk_ckpt_path}")
        
        print(f"\n--- Processing Sweep at Round {R} ---")
        print(f"Branching from end of Round {resume_round}")
        
        branch_start_round = int(resume_round + 1)
        for lam in lambdas:
            lam_str = str(lam).replace(".", "")
            exp_name = f"sweep_r{R}_lam_{lam_str}"
            
            # Branch
            branch_cmd = [
                sys.executable,
                str(branch_script),
                "--run_id", args.run_id,
                "--source_exp", args.trunk_exp,
                "--target_exps", exp_name,
                "--start_round", str(branch_start_round),
                "--force"
            ]
            subprocess.run(branch_cmd, check=True, stdout=subprocess.DEVNULL)
            
            # Run
            run_cmd = [
                sys.executable,
                str(main_script),
                "--run_id", args.run_id,
                "--experiment_name", exp_name,
                "--start", "resume",
                "--n_rounds", str(R + 1),
                "--skip-training-ckpt", str(trunk_ckpt_path)
            ]
            
            while len(procs) >= args.workers:
                for p in procs:
                    if p.poll() is not None:
                        procs.remove(p)
                time.sleep(1)
                
            print(f"Starting sweep branch: {exp_name}...")
            p = subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            procs.append(p)
            
    # Wait for all to finish
    for p in procs:
        p.wait()
            
    print("\n=== Sweep Completed ===")
    print(f"Results are saved in results/runs/{args.run_id}")

if __name__ == "__main__":
    main()
