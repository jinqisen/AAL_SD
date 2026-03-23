import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import glob
from scipy.stats import wasserstein_distance

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config
from core.dataset import Landslide4SenseDataset

def calculate_jaccard(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_minority_class_ratio(dataset, selected_indices):
    """Calculate the ratio of minority class (landslide, class 1) pixels in the selected samples."""
    total_pixels = 0
    minority_pixels = 0
    
    for idx in selected_indices:
        item = dataset[idx]
        mask = item['mask'].numpy()
        total_pixels += mask.size
        minority_pixels += np.sum(mask == 1) # Assuming 1 is landslide
        
    return minority_pixels / total_pixels if total_pixels > 0 else 0

def get_cluster_assignments(dataset):
    """Load pre-computed cluster assignments for the dataset."""
    cluster_file = "results/pools/lambda_sweep_experiment_01/fixed_lambda/unlabeled_pool_with_clusters.csv"
    # Alternative location if the above doesn't exist
    alt_file = "results/pools/cluster_experiment/unlabeled_pool_with_clusters.csv"
    
    import pandas as pd
    if os.path.exists(cluster_file):
        df = pd.read_csv(cluster_file)
    elif os.path.exists(alt_file):
        df = pd.read_csv(alt_file)
    else:
        # Check if any cluster file exists anywhere
        cluster_files = glob.glob("results/**/*cluster*.csv", recursive=True)
        if cluster_files:
            df = pd.read_csv(cluster_files[0])
        else:
            return None
            
    if 'sample_id' in df.columns and 'cluster' in df.columns:
        # Ensure sample_id is string and doesn't have 'image_' prefix for easy matching
        df['sample_id'] = df['sample_id'].astype(str).str.replace('image_', '')
        id_to_cluster = dict(zip(df['sample_id'], df['cluster']))
        return id_to_cluster
    return None

def calculate_cluster_coverage(selected_ids, id_to_cluster):
    """Calculate the number of unique clusters covered by the selected samples."""
    if not id_to_cluster:
        return 0
    covered_clusters = set()
    for sid in selected_ids:
        # Strip 'image_' prefix if it exists in the selected_ids but not in id_to_cluster
        clean_id = sid.replace('image_', '') if sid.startswith('image_') else sid
        
        # Try finding the exact id or the integer version
        if clean_id in id_to_cluster:
            covered_clusters.add(id_to_cluster[clean_id])
        else:
            try:
                int_id = str(int(clean_id))
                if int_id in id_to_cluster:
                    covered_clusters.add(id_to_cluster[int_id])
            except:
                pass
    return len(covered_clusters)

def get_cluster_distribution(selected_ids, id_to_cluster, total_clusters=20):
    """Get normalized distribution of samples across clusters."""
    if not id_to_cluster:
        return np.zeros(total_clusters)
        
    dist = np.zeros(total_clusters)
    count = 0
    for sid in selected_ids:
        clean_id = sid.replace('image_', '') if sid.startswith('image_') else sid
        c_id = None
        if clean_id in id_to_cluster:
            c_id = id_to_cluster[clean_id]
        else:
            try:
                int_id = str(int(clean_id))
                if int_id in id_to_cluster:
                    c_id = id_to_cluster[int_id]
            except:
                pass
                
        if c_id is not None and 0 <= c_id < total_clusters:
            dist[c_id] += 1
            count += 1
            
    if count > 0:
        dist = dist / count
    return dist

def load_trace(run_dir, strategy_name):
    trace_file = os.path.join(run_dir, f"{strategy_name}_trace.jsonl")
    if not os.path.exists(trace_file):
        return None
    
    selections = {}
    lambdas = {}
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'selection':
                    # Sometimes ids are strings, sometimes ints depending on trace format
                    selections[data['round']] = [str(x) for x in data['selected_ids']]
                    if 'lambda_val' in data and data['lambda_val'] is not None:
                        lambdas[data['round']] = data['lambda_val']
                    elif 'lambda' in data and data['lambda'] is not None:
                        lambdas[data['round']] = data['lambda']
                elif data.get('type') == 'lambda_controller_apply' or data.get('type') == 'lambda_policy_apply' or data.get('type') == 'lambda_override':
                    r = data.get('round')
                    l = data.get('lambda') or data.get('applied')
                    if r is not None and l is not None:
                        lambdas[r] = l
            except:
                continue
    return selections, lambdas

def main():
    cfg = Config()
    dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train")
    id_to_cluster = get_cluster_assignments(dataset)
    
    if not id_to_cluster:
        print("Warning: Cluster assignments not found. Cluster coverage and Wasserstein distance will be skipped.")
    else:
        total_clusters = max(id_to_cluster.values()) + 1
        print(f"Loaded cluster assignments for {len(id_to_cluster)} samples across {total_clusters} clusters.")
    
    # Create mapping from sample_id to dataset index
    id_to_idx = {}
    for i, img_name in enumerate(dataset.images):
        sample_id = os.path.splitext(img_name)[0]
        id_to_idx[sample_id] = i
        # Also map integer IDs if they exist in dataset as "image_X" or similar
        try:
            int_id = str(int(sample_id.replace('image_', '')))
            id_to_idx[int_id] = i
        except:
            pass
            
    # Target runs
    base_dir = "results/runs"
    seeds = ["42", "43", "44"]
    strategies = {
        "Random": "baseline_random",
        "Entropy": "baseline_entropy",
        "CoreSet": "baseline_coreset",
        "DIAL": "baseline_dial_style",
        "Wang": "baseline_wang_style",
        "Model A": "full_model_A_lambda_policy"
    }
    
    results = defaultdict(lambda: defaultdict(list))
    trajectories = defaultdict(lambda: defaultdict(dict))
    
    for seed in seeds:
        run_pattern = os.path.join(base_dir, f"baseline_20260309_*_seed{seed}")
        run_dirs = glob.glob(run_pattern)
        if not run_dirs:
            continue
        run_dir = run_dirs[0] # Pick the first match
        
        print(f"\nProcessing Seed {seed} from {run_dir}...")
        
        strategy_selections = {}
        for display_name, file_prefix in strategies.items():
            trace_data = load_trace(run_dir, file_prefix)
            if trace_data:
                selections, lambdas = trace_data
                strategy_selections[display_name] = selections
                if lambdas:
                    for rd, lam in lambdas.items():
                        trajectories[display_name][f"seed_{seed}"][rd] = lam
                print(f"  Loaded {len(selections)} rounds for {display_name}")
            else:
                print(f"  Missing trace for {display_name}")
                
        # Calculate Metrics for each strategy
        for strat_name, selections in strategy_selections.items():
            pass

    print("\n\n" + "="*50)
    print("ANALYSIS B1: LAMBDA TRAJECTORIES FOR ADAPTIVE STRATEGIES")
    print("="*50)
    for strat in ["DIAL", "Wang", "Model A"]:
        if strat in trajectories and trajectories[strat]:
            print(f"\n--- Strategy: {strat} ---")
            for seed, r_dict in trajectories[strat].items():
                lams = [f"{r_dict[rd]:.2f}" for rd in sorted(r_dict.keys())]
                print(f"{seed}: {' -> '.join(lams)}")
                # Calculate variance of lambda for this seed
                lam_vals = list(r_dict.values())
                if lam_vals:
                    print(f"  Lambda Std Dev: {np.std(lam_vals):.4f}")
                    
if __name__ == "__main__":
    main()
