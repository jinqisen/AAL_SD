import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.dataset import Landslide4SenseDataset
from core.model import LandslideDeepLabV3
from core.trainer import Trainer
from core.sampler import ADKUCSSampler

def get_base_pools():
    pool_dir = Path("results/pools/lambda_sweep_experiment_01/fixed_lambda")
    labeled_df = pd.read_csv(pool_dir / "labeled_pool.csv")
    labeled_df = labeled_df.iloc[:998]
    return labeled_df

def run_3seeds(target_ratio_str):
    cfg = Config()
    cfg.N_ROUNDS = 7
    cfg.EPOCHS_PER_ROUND = 10
    cfg.BATCH_SIZE = 16
    cfg.QUERY_SIZE = 221
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    cfg.DEVICE = device

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
            
    model_for_sampling = LandslideDeepLabV3(in_channels=cfg.IN_CHANNELS, classes=cfg.NUM_CLASSES)
    ckpt_path = "results/runs/lambda_sweep_experiment_01/fixed_lambda_round_models/round_05_best_val.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model_for_sampling.load_state_dict(state["state_dict"])
    model_for_sampling.to(device)
    model_for_sampling.eval()
    
    sampler = ADKUCSSampler(device=device, score_normalization=True)
    
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
    seeds = [42, 1024, 2024]
    
    if target_ratio_str == "1:1":
        bg_ratio = 1.0 / 42.14
    elif target_ratio_str == "10:1":
        bg_ratio = 10.0 / 42.14
    elif target_ratio_str == "42:1":
        bg_ratio = 1.0
    else:
        raise ValueError(f"Unknown ratio: {target_ratio_str}")
        
    print(f"Target ratio: {target_ratio_str}, using bg_undersample_ratio: {bg_ratio:.4f}")

    
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
        sel_indices = []
        for sid in selected_ids:
            for u_idx, orig_idx in enumerate(unlabeled_indices):
                if os.path.splitext(base_dataset.images[orig_idx])[0] == sid:
                    sel_indices.append(orig_idx)
                    break
        selected_indices_per_lambda[lam] = sel_indices

    results_all = {lam: [] for lam in lambdas}

    for seed in seeds:
        print(f"\n\n{'='*50}\nRUNNING SEED {seed}\n{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_dataset = Landslide4SenseDataset(cfg.DATA_DIR, split="train", bg_undersample_ratio=bg_ratio)
        
        for lam in lambdas:
            print(f"\n--- Seed {seed}, Lambda {lam} ---")
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
            print(f"Seed {seed}, Lambda {lam} Final Best mIoU: {best_miou:.4f}")
            results_all[lam].append(best_miou)

    print("\n\n" + "="*60)
    print(f"FINAL 3-SEED RESULTS FOR {target_ratio_str} IMBALANCE")
    print("="*60)
    avgs = {}
    for lam in lambdas:
        avg = np.mean(results_all[lam])
        std = np.std(results_all[lam])
        avgs[lam] = avg
        print(f"Lambda {lam:.1f}: {avg:.4f} ± {std:.4f} (Raw: {results_all[lam]})")
    
    max_avg = max(avgs.values())
    min_avg = min(avgs.values())
    print(f"\nAverage mIoU Range (极差): {max_avg - min_avg:.4f}")
    
    import json
    safe_ratio = target_ratio_str.replace(":", "_")
    with open(f"results_{safe_ratio}_3seeds.json", "w") as f:
        json.dump(results_all, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=str, choices=["1:1", "10:1", "42:1"], required=True, help="Target class imbalance ratio")
    args = parser.parse_args()
    run_3seeds(args.ratio)
