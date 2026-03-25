import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.dataset import Landslide4SenseDataset
from core.model import LandslideDeepLabV3
from core.sampler import ADKUCSSampler

def get_base_pools():
    pool_dir = Path("results/pools/lambda_sweep_experiment_01/fixed_lambda")
    labeled_df = pd.read_csv(pool_dir / "labeled_pool.csv")
    labeled_df = labeled_df.iloc[:998]
    return labeled_df

def run_cluster_analysis():
    cfg = Config()
    cfg.QUERY_SIZE = 221
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    
    dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train")
    
    labeled_df = get_base_pools()
    labeled_ids = set(labeled_df["sample_id"].tolist())
    
    labeled_indices = []
    unlabeled_indices = []
    for i, img_name in enumerate(dataset.images):
        sample_id = os.path.splitext(img_name)[0]
        if sample_id in labeled_ids:
            labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)
            
    model_for_sampling = LandslideDeepLabV3(in_channels=cfg.IN_CHANNELS, classes=cfg.NUM_CLASSES)
    ckpt_path = "results/runs/lambda_sweep_experiment_01/fixed_lambda_round_models/round_05_best_val.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model_for_sampling.load_state_dict(state["state_dict"])
    model_for_sampling.to(device)
    model_for_sampling.eval()
    
    sampler = ADKUCSSampler(device=device, score_normalization=True)
    
    loader_u = DataLoader(Subset(dataset, unlabeled_indices), batch_size=8, shuffle=False)
    loader_l = DataLoader(Subset(dataset, labeled_indices), batch_size=8, shuffle=False)
    
    _, features_l, _ = sampler.get_uncertainty_and_features(model_for_sampling, loader_l)
    labeled_features_np = features_l.numpy()
    
    u_scores_arr, features_u, _ = sampler.get_uncertainty_and_features(model_for_sampling, loader_u)
    features_u_np = features_u.numpy()
    
    unlabeled_info = {}
    for i, idx in enumerate(unlabeled_indices):
        unlabeled_info[idx] = {
            "uncertainty_score": u_scores_arr[i],
            "feature": features_u_np[i]
        }
        
    # We do a clustering of the entire pool just to see where they land
    pool_features = np.concatenate([labeled_features_np, features_u_np], axis=0)
    print("Clustering all features to find 'Cluster 19'...")
    kmeans = KMeans(n_clusters=88, random_state=42, n_init="auto")
    pool_labels = kmeans.fit_predict(pool_features)
    
    # Identify the largest cluster
    unique, counts = np.unique(pool_labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    print(f"Largest cluster is {largest_cluster} with {counts[np.argmax(counts)]} samples out of {len(pool_features)}")
    
    # Let's map each unlabeled index to its cluster
    # pool_features = [labeled..., unlabeled...]
    # So unlabeled features start at len(labeled_features_np)
    unlabeled_cluster_labels = pool_labels[len(labeled_features_np):]
    
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for lam in lambdas:
        ranked = sampler.rank_samples(
            unlabeled_info=unlabeled_info,
            labeled_features=labeled_features_np,
            current_iteration=5,
            total_iterations=7,
            lambda_override=lam
        )
        
        selected_ids = [r["sample_id"] for r in ranked[:cfg.QUERY_SIZE]]
        
        # Find which indices in unlabeled_indices were selected
        selected_indices_in_unlabeled = []
        for sid in selected_ids:
            # sid is the orig_idx from unlabeled_indices
            u_idx = unlabeled_indices.index(sid)
            selected_indices_in_unlabeled.append(u_idx)
                    
        selected_clusters = unlabeled_cluster_labels[selected_indices_in_unlabeled]
        sel_unique, sel_counts = np.unique(selected_clusters, return_counts=True)
        
        # Sort by count
        sorted_clusters = sorted(zip(sel_unique, sel_counts), key=lambda x: x[1], reverse=True)
        top_clusters_str = ", ".join([f"C{c}({cnt})" for c, cnt in sorted_clusters[:5]])
        
        # How many in the largest cluster?
        count_in_largest = sum(cnt for c, cnt in sorted_clusters if c == largest_cluster)
        
        print(f"Lambda {lam:.1f} | Samples in Largest Cluster (C{largest_cluster}): {count_in_largest}/{cfg.QUERY_SIZE} ({(count_in_largest/cfg.QUERY_SIZE)*100:.1f}%) | Top 5 clusters: {top_clusters_str}")

if __name__ == "__main__":
    run_cluster_analysis()
