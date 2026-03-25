from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import h5py

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import Config
from core.model import LandslideDeepLabV3

FIG_DIR = Path(__file__).resolve().parents[1].parent / "paper_draft" / "figures"

METHODS_DEFAULT = [
    "full_model_A_lambda_policy",
    "baseline_entropy",
    "baseline_coreset",
    "baseline_random",
]

METHOD_LABELS = {
    "full_model_A_lambda_policy": "AAL-SD (A)",
    "full_model_B_lambda_agent": "AAL-SD (B)",
    "no_risk_control": "w/o Risk Ctrl",
    "baseline_entropy": "Entropy",
    "baseline_coreset": "Core-Set",
    "baseline_random": "Random",
    "baseline_bald": "BALD",
    "baseline_wang_style": "Wang-style",
    "baseline_dial_style": "DIAL-style",
}


def _find_checkpoint(
    checkpoint_dir: str, run_id: str, experiment: str, target_round: int | None = None
) -> str | None:
    base = os.path.join(checkpoint_dir, run_id)
    if not os.path.isdir(base):
        return None
    patterns = [
        f"{experiment}_state.json",
        f"{experiment}_round*_best.pth",
        f"{experiment}_best.pth",
        f"{experiment}_final.pth",
        f"{experiment}*.pth",
    ]
    if target_round is not None:
        patterns.insert(0, f"{experiment}_round{target_round}_best.pth")
        patterns.insert(1, f"{experiment}_round{target_round}.pth")

    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(base, pat)))
        if matches:
            for m in reversed(matches):
                if m.endswith(".pth"):
                    return m
    pth_files = sorted(
        glob.glob(os.path.join(base, "**", f"*{experiment}*.pth"), recursive=True)
    )
    return pth_files[-1] if pth_files else None


def _load_model(checkpoint_path: str, device: str) -> LandslideDeepLabV3:
    model = LandslideDeepLabV3(in_channels=14, classes=2)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_test_sample(data_dir: str, idx: int):
    img_dir = os.path.join(data_dir, "TestData", "img")
    mask_dir = os.path.join(data_dir, "TestData", "mask")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".h5")])
    if idx >= len(img_files):
        raise IndexError(f"Test index {idx} out of range (max {len(img_files) - 1})")
    img_path = os.path.join(img_dir, img_files[idx])
    with h5py.File(img_path, "r") as f:
        keys = list(f.keys())
        image = np.array(f[keys[0]], dtype=np.float32)

    mask = None
    if os.path.isdir(mask_dir):
        img_id = os.path.splitext(img_files[idx])[0]
        for candidate in [
            img_id + ".h5",
            "mask_" + img_id.replace("image_", "") + ".h5",
        ]:
            mp = os.path.join(mask_dir, candidate)
            if os.path.exists(mp):
                with h5py.File(mp, "r") as f:
                    keys = list(f.keys())
                    mask = np.array(f[keys[0]], dtype=np.int64)
                break
    return image, mask, img_files[idx]


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] > image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] >= 4:
        rgb = image[..., [3, 2, 1]].copy()
    else:
        rgb = image[..., :3].copy()
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        if hi - lo > 1e-6:
            rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
        else:
            rgb[..., c] = 0.0
    return rgb.astype(np.float32)


def _predict(model: LandslideDeepLabV3, image: np.ndarray, device: str) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(image).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor.to(device))
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()


def _overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45
) -> np.ndarray:
    out = rgb.copy()
    for c in range(3):
        out[..., c] = np.where(
            mask > 0, out[..., c] * (1 - alpha) + color[c] * alpha, out[..., c]
        )
    return np.clip(out, 0, 1)


def _compute_iou(pred: np.ndarray, gt: np.ndarray, cls: int = 1) -> float:
    intersection = np.sum((pred == cls) & (gt == cls))
    union = np.sum((pred == cls) | (gt == cls))
    return float(intersection / union) if union > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument(
        "--test_indices", type=int, nargs="+", default=[0, 50, 100, 200]
    )
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    methods = args.methods or METHODS_DEFAULT
    test_indices = args.test_indices
    output_path = (
        Path(args.output) if args.output else FIG_DIR / "Fig12_Segmentation_Viz.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = {}
    for method in methods:
        ckpt = _find_checkpoint(cfg.CHECKPOINT_DIR, args.run_id, method, args.round)
        if ckpt is None:
            print(
                f"WARNING: No checkpoint found for {method} in {args.run_id}, skipping"
            )
            continue
        print(f"Loading {method}: {ckpt}")
        models[method] = _load_model(ckpt, device)

    if not models:
        print("ERROR: No checkpoints found. Run experiments first.")
        return 1

    n_rows = len(test_indices)
    n_cols = 2 + len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    gt_color = (0.0, 0.8, 0.0)
    pred_color = (0.9, 0.1, 0.1)

    for row, tidx in enumerate(test_indices):
        image, gt_mask, fname = _load_test_sample(cfg.DATA_DIR, tidx)
        rgb = _to_rgb(image)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Input RGB" if row == 0 else "", fontsize=9)
        axes[row, 0].set_ylabel(f"#{tidx}", fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        if gt_mask is not None:
            gt_viz = _overlay_mask(rgb, gt_mask, gt_color)
            axes[row, 1].imshow(gt_viz)
        else:
            axes[row, 1].imshow(rgb)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "", fontsize=9)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        for col_offset, method in enumerate(models.keys()):
            pred = _predict(models[method], image, device)
            pred_viz = _overlay_mask(rgb, pred, pred_color)
            ax = axes[row, 2 + col_offset]
            ax.imshow(pred_viz)
            label = METHOD_LABELS.get(method, method)
            if gt_mask is not None:
                iou = _compute_iou(pred, gt_mask)
                title = f"{label}\nIoU={iou:.3f}" if row == 0 else f"IoU={iou:.3f}"
            else:
                title = label if row == 0 else ""
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
