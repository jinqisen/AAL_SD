import json
import os
import glob
import numpy as np

def load_selected_ids(trace_file):
    selected_ids_by_round = {}
    if not os.path.exists(trace_file):
        return selected_ids_by_round
    with open(trace_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if data.get("type") == "selection":
                    rnd = data.get("round")
                    selected_ids = data.get("selected_ids", [])
                    if "selection" in data and "selected_ids" in data["selection"]:
                        selected_ids = data["selection"]["selected_ids"]
                    if rnd is not None and selected_ids:
                        selected_ids_by_round[rnd] = set(selected_ids)
            except Exception as e:
                pass
    return selected_ids_by_round

def calculate_jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def analyze_seed_overlap(seed_dir):
    strategies = [
        "uncertainty_only",
        "knowledge_only",
        "fixed_lambda",
        "full_model_A_lambda_policy",
        "no_agent"
    ]
    
    data = {}
    for strat in strategies:
        trace_path = os.path.join(seed_dir, f"{strat}_trace.jsonl")
        data[strat] = load_selected_ids(trace_path)
        
    print(f"\n--- Analysis for {os.path.basename(seed_dir)} ---")
    
    rounds = set()
    for strat, ids in data.items():
        rounds.update(ids.keys())
    
    for rnd in sorted(list(rounds)):
        print(f"Round {rnd}:")
        u_ids = data.get("uncertainty_only", {}).get(rnd, set())
        k_ids = data.get("knowledge_only", {}).get(rnd, set())
        fixed_ids = data.get("fixed_lambda", {}).get(rnd, set())
        full_ids = data.get("full_model_A_lambda_policy", {}).get(rnd, set())
        
        if u_ids and k_ids:
            j_uk = calculate_jaccard(u_ids, k_ids)
            print(f"  Uncertainty vs Knowledge Jaccard: {j_uk:.1%} (Overlap: {len(u_ids.intersection(k_ids))} / {len(u_ids)})")
        if u_ids and fixed_ids:
            j_uf = calculate_jaccard(u_ids, fixed_ids)
            print(f"  Uncertainty vs Fixed Jaccard: {j_uf:.1%}")
        if u_ids and full_ids:
            j_ua = calculate_jaccard(u_ids, full_ids)
            print(f"  Uncertainty vs Full Agent Jaccard: {j_ua:.1%}")

def analyze_final_miou(seed_dir):
    exp_file = os.path.join(seed_dir, "experiment_results.json")
    if not os.path.exists(exp_file):
        return
    
    with open(exp_file, 'r') as f:
        results = json.load(f)
        
    print(f"\nFinal mIoU for {os.path.basename(seed_dir)}:")
    for strat, res in results.items():
        if isinstance(res, dict) and "performance_history" in res:
            history = res["performance_history"]
            if history:
                miou = history[-1].get("mIoU", 0)
                print(f"  {strat}: {miou:.4f}")

if __name__ == "__main__":
    base_dir = "/Users/anykong/AD-KUCS/AAL_SD/results/runs"
    seeds = glob.glob(os.path.join(base_dir, "ablation_matrix_p3_20260323_235540_seed*"))
    for seed_dir in sorted(seeds):
        if os.path.isdir(seed_dir):
            analyze_seed_overlap(seed_dir)
            analyze_final_miou(seed_dir)
