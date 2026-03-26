import json
import os
import glob

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

if __name__ == "__main__":
    base_dir = "/Users/anykong/AD-KUCS/AAL_SD/results/runs"
    seeds = glob.glob(os.path.join(base_dir, "baseline_20260309_020108_seed*"))
    for seed_dir in sorted(seeds):
        if os.path.isdir(seed_dir):
            print(f"\n--- Analysis for {os.path.basename(seed_dir)} ---")
            trace_ent = os.path.join(seed_dir, "baseline_entropy_trace.jsonl")
            trace_core = os.path.join(seed_dir, "baseline_coreset_trace.jsonl")
            
            ent_ids = load_selected_ids(trace_ent)
            core_ids = load_selected_ids(trace_core)
            
            rounds = sorted(list(set(ent_ids.keys()).union(core_ids.keys())))
            for rnd in rounds:
                e = ent_ids.get(rnd, set())
                c = core_ids.get(rnd, set())
                if e and c:
                    print(f"Round {rnd} Ent vs Core Jaccard: {calculate_jaccard(e, c):.1%}")
