
import os
import json
import shutil
from pathlib import Path
import glob

# Configuration
BASE_DIR = "/Users/anykong/AAL_SD"
RESULTS_RUNS_DIR = os.path.join(BASE_DIR, "results/runs")
RESULTS_CKPT_DIR = os.path.join(BASE_DIR, "results/checkpoints")
DATA_POOLS_DIR = os.path.join(BASE_DIR, "results/pools")

TARGET_RUN_ID = "20260204_111154_strict"
SOURCE_RUN_IDS = [
    "20260204_111154_strict", # Include target itself in comparison
    "20260204_135725_strict",
    "20260204_183956_strict",
    "20260204_194812_strict"
]

def get_experiment_status(run_id, exp_name):
    """
    Reads the status json for a given experiment in a specific run.
    Returns (round, updated_at, status_data)
    """
    status_path = os.path.join(RESULTS_RUNS_DIR, run_id, f"{exp_name}_status.json")
    if not os.path.exists(status_path):
        return -1, "", None
    
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        
        # Extract progress round
        current_round = data.get("progress", {}).get("round", 0)
        updated_at = data.get("updated_at", "")
        
        return current_round, updated_at, data
    except Exception as e:
        print(f"Error reading status for {run_id}/{exp_name}: {e}")
        return -1, "", None

def get_all_experiments():
    """Finds all unique experiment names across all source run IDs."""
    experiments = set()
    for run_id in SOURCE_RUN_IDS:
        run_dir = os.path.join(RESULTS_RUNS_DIR, run_id)
        if not os.path.exists(run_dir):
            continue
        
        # Look for _status.json files
        for f in os.listdir(run_dir):
            if f.endswith("_status.json"):
                exp_name = f.replace("_status.json", "")
                experiments.add(exp_name)
    return sorted(list(experiments))

def patch_status_file(status_path, new_run_id):
    """Updates paths and run_id in the copied status file."""
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        
        data["run_id"] = new_run_id
        
        # Update pools_dir
        if "pools_dir" in data:
            # Replace old run_id in path with new run_id
            # Assuming path structure: .../results/pools/{OLD_ID}/{EXP_NAME}
            old_pools_dir = data["pools_dir"]
            # A simple string replace might be risky if run_id is a substring of something else, 
            # but given the timestamp format, it's likely unique.
            # Safer: construct new path
            exp_name = data["experiment_name"]
            new_pools_dir = os.path.join(DATA_POOLS_DIR, new_run_id, exp_name)
            data["pools_dir"] = new_pools_dir

        # Update checkpoint_path
        if "checkpoint_path" in data:
            # Structure: .../results/checkpoints/{OLD_ID}/{EXP_NAME}_state.json
            exp_name = data["experiment_name"]
            new_ckpt_path = os.path.join(RESULTS_CKPT_DIR, new_run_id, f"{exp_name}_state.json")
            data["checkpoint_path"] = new_ckpt_path
            
        with open(status_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Failed to patch status file {status_path}: {e}")

def merge_experiments():
    experiments = get_all_experiments()
    print(f"Found {len(experiments)} unique experiments: {experiments}")
    
    for exp_name in experiments:
        best_run_id = None
        max_round = -1
        best_updated_at = ""
        
        # 1. Find the best run
        for run_id in SOURCE_RUN_IDS:
            rnd, updated_at, _ = get_experiment_status(run_id, exp_name)
            if rnd > max_round:
                max_round = rnd
                best_run_id = run_id
                best_updated_at = updated_at
            elif rnd == max_round and max_round != -1:
                # Tie-breaker: use latest timestamp
                if updated_at > best_updated_at:
                    best_run_id = run_id
                    best_updated_at = updated_at
        
        if not best_run_id:
            print(f"Skipping {exp_name}: No valid status found.")
            continue
            
        print(f"[{exp_name}] Best Run: {best_run_id} (Round {max_round})")
        
        # If the best run is already the target, we don't need to copy anything
        # UNLESS the target directory is missing some files that are present in the best run (which is itself).
        # But here 'target' is in 'source_ids', so if best_run_id == TARGET_RUN_ID, we are good.
        if best_run_id == TARGET_RUN_ID:
            print(f"  -> Already in target. No action needed.")
            continue
            
        # 2. Perform Copy
        print(f"  -> Merging from {best_run_id} to {TARGET_RUN_ID}...")
        
        # Ensure target directories exist
        target_run_dir = os.path.join(RESULTS_RUNS_DIR, TARGET_RUN_ID)
        target_ckpt_dir = os.path.join(RESULTS_CKPT_DIR, TARGET_RUN_ID)
        target_pool_dir = os.path.join(DATA_POOLS_DIR, TARGET_RUN_ID, exp_name)
        
        os.makedirs(target_run_dir, exist_ok=True)
        os.makedirs(target_ckpt_dir, exist_ok=True)
        # Pool dir is created during copytree usually, or we ensure parent exists
        os.makedirs(os.path.dirname(target_pool_dir), exist_ok=True)
        
        # A. Copy Run Logs/Status (Overwrite)
        src_run_dir = os.path.join(RESULTS_RUNS_DIR, best_run_id)
        for f in os.listdir(src_run_dir):
            if f.startswith(exp_name):
                src_file = os.path.join(src_run_dir, f)
                dst_file = os.path.join(target_run_dir, f)
                shutil.copy2(src_file, dst_file)
        
        # B. Copy Checkpoints (Overwrite)
        src_ckpt_dir = os.path.join(RESULTS_CKPT_DIR, best_run_id)
        if os.path.exists(src_ckpt_dir):
            for f in os.listdir(src_ckpt_dir):
                if f.startswith(exp_name):
                    src_file = os.path.join(src_ckpt_dir, f)
                    dst_file = os.path.join(target_ckpt_dir, f)
                    shutil.copy2(src_file, dst_file)
        
        # C. Copy Pools (Overwrite directory)
        src_pool_dir = os.path.join(DATA_POOLS_DIR, best_run_id, exp_name)
        if os.path.exists(src_pool_dir):
            if os.path.exists(target_pool_dir):
                shutil.rmtree(target_pool_dir)
            shutil.copytree(src_pool_dir, target_pool_dir)
            
        # 3. Patch Status File
        target_status_path = os.path.join(target_run_dir, f"{exp_name}_status.json")
        if os.path.exists(target_status_path):
            patch_status_file(target_status_path, TARGET_RUN_ID)
            print(f"  -> Patched status file.")

def cleanup_old_runs():
    """Deletes source run directories that are not the target run."""
    print("\nStarting cleanup of old runs...")
    for run_id in SOURCE_RUN_IDS:
        if run_id == TARGET_RUN_ID:
            continue
            
        print(f"Removing data for {run_id}...")
        
        # 1. Remove results/runs
        run_dir = os.path.join(RESULTS_RUNS_DIR, run_id)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f"  - Deleted {run_dir}")
            
        # 2. Remove results/checkpoints
        ckpt_dir = os.path.join(RESULTS_CKPT_DIR, run_id)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            print(f"  - Deleted {ckpt_dir}")
            
        # 3. Remove results/pools
        pool_dir = os.path.join(DATA_POOLS_DIR, run_id)
        if os.path.exists(pool_dir):
            shutil.rmtree(pool_dir)
            print(f"  - Deleted {pool_dir}")

if __name__ == "__main__":
    merge_experiments()
    cleanup_old_runs()
