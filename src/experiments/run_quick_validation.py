import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.dataset import Landslide4SenseDataset
from core.model import LandslideDeepLabV3
from core.trainer import Trainer
from core.sampler import ADKUCSSampler
from utils.logger import logger

def get_base_pools():
    pool_dir = Path("results/pools/lambda_sweep_experiment_01/fixed_lambda")
    labeled_df = pd.read_csv(pool_dir / "labeled_pool.csv")
    unlabeled_df = pd.read_csv(pool_dir / "unlabeled_pool.csv")
    
    # In round 5, labeled size was 998
    labeled_df = labeled_df.iloc[:998]
    
    # Reconstruct unlabeled pool
    all_unlabeled = pd.read_csv(pool_dir / "unlabeled_pool.csv")
    # Actually, the unlabeled pool for round 5 is all samples minus the 998 labeled samples.
    # It's easier to just take the full dataset and subtract labeled.
    return labeled_df

def run_validation():
    cfg = Config()
    cfg.N_ROUNDS = 7
    cfg.EPOCHS_PER_ROUND = 10
    cfg.BATCH_SIZE = 16
    cfg.QUERY_SIZE = 221
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    cfg.DEVICE = device

    dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train")
    val_dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="val")
    
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    labeled_df = get_base_pools()
    labeled_ids = set(labeled_df["sample_id"].tolist())
    
    # Map filenames to indices
    labeled_indices = []
    unlabeled_indices = []
    for i, img_name in enumerate(dataset.images):
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
    loader_u = DataLoader(Subset(dataset, unlabeled_indices), batch_size=8, shuffle=False)
    loader_l = DataLoader(Subset(dataset, labeled_indices), batch_size=8, shuffle=False)
    
    # Extract labeled features
    _, features_l, _ = sampler.get_uncertainty_and_features(model_for_sampling, loader_l)
    labeled_features_np = features_l.numpy()
    
    # Extract unlabeled features and U scores
    u_scores_arr, features_u, _ = sampler.get_uncertainty_and_features(model_for_sampling, loader_u)
    features_u_np = features_u.numpy()
    
    unlabeled_info = {}
    for i, idx in enumerate(unlabeled_indices):
        unlabeled_info[idx] = {
            "uncertainty_score": u_scores_arr[i],
            "feature": features_u_np[i]
        }
        
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    for lam in lambdas:
        print(f"\n{'='*50}\nTesting lambda = {lam}\n{'='*50}")
        ranked = sampler.rank_samples(
            unlabeled_info=unlabeled_info,
            labeled_features=labeled_features_np,
            current_iteration=5,
            total_iterations=7,
            lambda_override=lam
        )
        
        selected_ids = [r["sample_id"] for r in ranked[:cfg.QUERY_SIZE]]
        print(f"Selected {len(selected_ids)} samples.")
        
        # Train new model
        current_labeled = labeled_indices + selected_ids
        train_loader = DataLoader(Subset(dataset, current_labeled), batch_size=cfg.BATCH_SIZE, shuffle=True)
        
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
            
        results[lam] = best_miou
        print(f"Lambda {lam} Final Best mIoU: {best_miou:.4f}")
        
    print("\n\n" + "="*50)
    print("FINAL SWEEP RESULTS")
    print("="*50)
    for lam, miou in results.items():
        print(f"Lambda {lam:.1f}: {miou:.4f}")
        
    max_miou = max(results.values())
    min_miou = min(results.values())
    print(f"mIoU Range (极差): {max_miou - min_miou:.4f}")

if __name__ == "__main__":
    run_validation()
