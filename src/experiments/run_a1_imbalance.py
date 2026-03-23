import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.dataset import Landslide4SenseDataset
from core.model import LandslideDeepLabV3
from core.trainer import Trainer
from core.sampler import ADKUCSSampler

def get_base_pools():
    pool_dir = Path("results/pools/lambda_sweep_experiment_01/fixed_lambda")
    labeled_df = pd.read_csv(pool_dir / "labeled_pool.csv")
    # Take first 998 to simulate Round 5 base
    labeled_df = labeled_df.iloc[:998]
    return labeled_df

def run_a1_experiment():
    cfg = Config()
    cfg.N_ROUNDS = 7
    cfg.EPOCHS_PER_ROUND = 10
    cfg.BATCH_SIZE = 16
    cfg.QUERY_SIZE = 221
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    cfg.DEVICE = device

    # We will instantiate dataset inside the loop with different bg_undersample_ratio
    # But for sampling, we use the standard dataset
    base_dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train", bg_undersample_ratio=1.0)
    val_dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    labeled_df = get_base_pools()
    labeled_ids = set(labeled_df["sample_id"].tolist())
    
    labeled_indices = []
    unlabeled_indices = []
    for i, img_name in enumerate(base_dataset.images):
        sample_id = os.path.splitext(img_name)[0]
        if sample_id in labeled_ids:
            labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)
            
    print(f"Base pool: Labeled={len(labeled_indices)}, Unlabeled={len(unlabeled_indices)}")
    
    # Load Round 5 model for sampling
    model_for_sampling = LandslideDeepLabV3(in_channels=cfg.IN_CHANNELS, classes=cfg.NUM_CLASSES)
    ckpt_path = "results/runs/lambda_sweep_experiment_01/fixed_lambda_round_models/round_05_best_val.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model_for_sampling.load_state_dict(state["state_dict"])
    model_for_sampling.to(device)
    model_for_sampling.eval()
    
    sampler = ADKUCSSampler(device=device, score_normalization=True)
    
    print("Extracting features and uncertainties...")
    loader_u = DataLoader(Subset(base_dataset, unlabeled_indices), batch_size=8, shuffle=False)
    loader_l = DataLoader(Subset(base_dataset, labeled_indices), batch_size=8, shuffle=False)
    
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
        
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # We want: 10:1, 1:1 (skip Original as we already have it)
    ratios = {
        "1:1": 1.0 / 42.14,
        "10:1": 10.0 / 42.14
    }
    
    # Pre-calculate ranked samples so we use EXACTLY the same samples for a given lambda across all ratios
    # Wait, the sampler depends on unlabeled_info which is static. So for a given lambda, the selected samples are static.
    selected_indices_per_lambda = {}
    for lam in lambdas:
        ranked = sampler.rank_samples(
            unlabeled_info=unlabeled_info,
            labeled_features=labeled_features_np,
            current_iteration=5,
            total_iterations=7,
            lambda_override=lam
        )
        selected_ids = [r["sample_id"] for r in ranked[:cfg.QUERY_SIZE]]
        
        # map back to indices
        sel_indices = []
        for sid in selected_ids:
            for u_idx, orig_idx in enumerate(unlabeled_indices):
                if os.path.splitext(base_dataset.images[orig_idx])[0] == sid:
                    sel_indices.append(orig_idx)
                    break
        selected_indices_per_lambda[lam] = sel_indices
        
    all_results = {}
    
    for ratio_name, bg_ratio in ratios.items():
        print(f"\n\n{'#'*60}\nRunning Imbalance: {ratio_name} (bg_undersample_ratio={bg_ratio:.4f})\n{'#'*60}")
        
        # Create dataset for this ratio
        train_dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train", bg_undersample_ratio=bg_ratio)
        
        ratio_results = {}
        for lam in lambdas:
            print(f"\n{'-'*40}\nTesting lambda = {lam}\n{'-'*40}")
            current_labeled = labeled_indices + selected_indices_per_lambda[lam]
            train_loader = DataLoader(Subset(train_dataset, current_labeled), batch_size=cfg.BATCH_SIZE, shuffle=True)
            
            model = LandslideDeepLabV3(in_channels=cfg.IN_CHANNELS, classes=cfg.NUM_CLASSES)
            trainer = Trainer(model, cfg, device)
            
            best_miou = 0.0
            for epoch in range(cfg.EPOCHS_PER_ROUND):
                loss = trainer.train_one_epoch(train_loader)
                if isinstance(loss, tuple): loss = loss[0]
                metrics = trainer.evaluate(val_loader)
                miou = metrics['mIoU']
                if miou > best_miou:
                    best_miou = miou
                print(f"Epoch {epoch+1}/{cfg.EPOCHS_PER_ROUND} - Loss: {loss:.4f}, mIoU: {miou:.4f} (Best: {best_miou:.4f})")
                
            ratio_results[lam] = best_miou
            print(f"[{ratio_name}] Lambda {lam} Final Best mIoU: {best_miou:.4f}")
            
        all_results[ratio_name] = ratio_results
        
    print("\n\n" + "="*60)
    print("FINAL A1 EXPERIMENT RESULTS")
    print("="*60)
    for ratio_name, results in all_results.items():
        print(f"\n{ratio_name}:")
        for lam, miou in results.items():
            print(f"  Lambda {lam:.1f}: {miou:.4f}")
        max_miou = max(results.values())
        min_miou = min(results.values())
        print(f"  mIoU Range (极差): {max_miou - min_miou:.4f}")
        
        # Save to file for further analysis
        import json
        with open("a1_results.json", "w") as f:
            json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    run_a1_experiment()
