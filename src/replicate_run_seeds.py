import os
import sys
import glob
from pathlib import Path

# Add src to path just in case
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from experiments.run_multi_seed import main as run_multi_seed_main
except ImportError:
    # Try adding current directory to path if running from src
    sys.path.append(os.getcwd())
    from experiments.run_multi_seed import main as run_multi_seed_main

def get_experiments(run_dir):
    experiments = []
    # Use glob to find status files
    status_files = glob.glob(os.path.join(run_dir, "*_status.json"))
    if not status_files:
        print(f"Warning: No status files found in {run_dir}")
        return []
        
    for f in status_files:
        name = os.path.basename(f).replace("_status.json", "")
        experiments.append(name)
    return sorted(experiments)

def replicate_seeds():
    # Source run path from user input
    source_run_path = "/Users/anykong/AAL_SD/results/runs/run_src_full_model_with_baselines_seed42"
    
    if not os.path.exists(source_run_path):
        print(f"Error: {source_run_path} not found")
        return

    experiments = get_experiments(source_run_path)
    if not experiments:
        print("No experiments found to replicate.")
        return
        
    print(f"Found {len(experiments)} experiments: {experiments}")

    # Base run ID derived from source run name
    # The source is run_src_full_model_with_baselines_seed42
    # We want base_run_id to be run_src_full_model_with_baselines
    base_run_id = "run_src_full_model_with_baselines"
    
    # Seeds: 43, 44, 45, 46
    seeds = "43 44 45 46"
    
    # Arguments for run_multi_seed
    # Note: run_multi_seed.main takes a list of strings (argv)
    argv = [
        "--run_id", base_run_id,
        "--seeds", seeds,
        "--experiments", *experiments,
        "--parallel",
        "--workers", "4",
        "--results_dir", "/Users/anykong/AAL_SD/results"
    ]
    
    print(f"Starting multi-seed run for {base_run_id} with seeds {seeds}")
    
    try:
        # run_multi_seed.main expects argv, usually without the script name at argv[0] if called directly,
        # but argparse.parse_args(argv) uses argv as is. 
        # If passed explicitly, argparse uses the list provided.
        run_multi_seed_main(argv)
    except SystemExit as e:
        print(f"run_multi_seed finished with code {e.code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    replicate_seeds()
