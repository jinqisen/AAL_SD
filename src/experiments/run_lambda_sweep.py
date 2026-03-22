import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run lambda sweep at a specific round")
    parser.add_argument("--run-id", required=True, help="Run ID of the trunk experiment")
    parser.add_argument("--trunk-exp", required=True, help="Name of the trunk experiment")
    parser.add_argument("--sweep-round", type=int, required=True, help="Round to perform sweep ON (e.g. 6). This means it branches after Round 5, selects for Round 6, and trains Round 6.")
    parser.add_argument("--lambdas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", help="Comma-separated lambdas")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    lambdas = [float(x) for x in args.lambdas.split(",")]
    
    K = args.sweep_round
    resume_round = K - 2
    if resume_round < 0:
        raise ValueError("Sweep round must be >= 2")
        
    trunk_ckpt_path = base_dir / f"results/runs/{args.run_id}/{args.trunk_exp}_round_models/round_{K-1:02d}_best_val.pt"
    if not trunk_ckpt_path.exists():
        raise FileNotFoundError(f"Trunk checkpoint not found: {trunk_ckpt_path}")

    print(f"=== Preparing Lambda Sweep at Round {K} ===")
    print(f"Trunk: {args.run_id}/{args.trunk_exp}")
    print(f"Branching from end of Round {resume_round}")
    print(f"Skipping training for Round {K-1}, using checkpoint: {trunk_ckpt_path}")
    
    # 1. Update ablation_config.py
    config_path = base_dir / "src/experiments/ablation_config.py"
    config_content = config_path.read_text()
    
    new_configs = []
    exp_names = []
    for lam in lambdas:
        lam_str = str(lam).replace(".", "")
        exp_name = f"sweep_r{K}_lam_{lam_str}"
        exp_names.append(exp_name)
        if f'"{exp_name}":' not in config_content:
            new_configs.append(f"""
    "{exp_name}": {{
        "description": "Lambda sweep at Round {K}, lambda={lam}",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": {lam},
    }},""")
            
    if new_configs:
        print("Injecting new configs into ablation_config.py...")
        # find the last closing brace
        last_brace_idx = config_content.rfind('}')
        if last_brace_idx != -1:
            config_content = config_content[:last_brace_idx] + "".join(new_configs) + "\n}"
            config_path.write_text(config_content)
        else:
            raise RuntimeError("Could not parse ablation_config.py")

    # 2. Branch experiments
    branch_script = base_dir / "src/utils/branch_experiment_from_round.py"
    print("\nCreating branches...")
    for exp_name in exp_names:
        cmd = [
            sys.executable,
            str(branch_script),
            "--run_id", args.run_id,
            "--source_exp", args.trunk_exp,
            "--target_exps", exp_name,
            "--start_round", str(resume_round),
            "--force"
        ]
        subprocess.run(cmd, check=True)

    # 3. Run parallel strict
    run_parallel_script = base_dir / "src/run_parallel_strict.py"
    include_str = ",".join(exp_names)
    print("\nRunning parallel experiments...")
    
    # We need to pass skip_training_ckpt to each run. We can modify run_parallel_strict to accept it?
    # No, run_parallel_strict does not accept skip_training_ckpt. We can run main.py sequentially or update run_parallel_strict.
    # Since it's only 1 round of training per lambda, maybe just run them sequentially?
    # Or just use subprocess Popen for parallel.
    import time
    procs = []
    for exp_name in exp_names:
        cmd = [
            sys.executable,
            str(base_dir / "src/main.py"),
            "--run_id", args.run_id,
            "--experiment_name", exp_name,
            "--start", "resume",
            "--n_rounds", str(K + 1),
            "--skip-training-ckpt", str(trunk_ckpt_path)
        ]
        
        if args.workers == 1:
            print(f"Running {exp_name}...")
            subprocess.run(cmd, check=True)
        else:
            while len(procs) >= args.workers:
                for p in procs:
                    if p.poll() is not None:
                        procs.remove(p)
                time.sleep(1)
            print(f"Starting {exp_name}...")
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            procs.append(p)
            
    if procs:
        for p in procs:
            p.wait()
            
    print("\n=== Sweep Completed ===")
    print(f"Results are saved in results/runs/{args.run_id}")

if __name__ == "__main__":
    main()
