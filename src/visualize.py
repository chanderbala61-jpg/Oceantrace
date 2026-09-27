"""
Visualization Utilities for OceanTrace SegFormer Experiment 1
--------------------------------------------------------------
Generates side-by-side comparison plots and transparent overlays:
  1. SAR Image (Grayscale dB)
  2. Ground-Truth Oil Mask (Green)
  3. Predicted Oil Mask (Red)
  4. Overlay (Green = True Positive, Red = False Positive, Blue = False Negative)
"""

import os
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


def create_prediction_figure(
    sar_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    threshold: float = 0.5,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Creates a 4-panel visual comparison.
    """
    pred_binary = (pred_prob >= threshold).astype(np.float32)
    gt_binary = (gt_mask >= 0.5).astype(np.float32)

    # Build RGB overlay:
    # Green = TP (Correct detection)
    # Red = FP (False alarm)
    # Yellow/Blue = FN (Missed oil)
    h, w = sar_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    # Base grayscale image in [0, 1]
    sar_norm = (np.clip(sar_img, -35.0, 5.0) - (-35.0)) / 40.0
    overlay[:, :, 0] = sar_norm
    overlay[:, :, 1] = sar_norm
    overlay[:, :, 2] = sar_norm

    # Highlights
    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)

    # TP in Green
    overlay[tp] = [0.0, 1.0, 0.0]
    # FP in Red
    overlay[fp] = [1.0, 0.2, 0.2]
    # FN in Cyan/Blue
    overlay[fn] = [0.2, 0.6, 1.0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    
    # 1. SAR
    axes[0].imshow(sar_img, cmap="gray", vmin=-35.0, vmax=5.0)
    axes[0].set_title("1. Sentinel-1 SAR ($\sigma^0$ dB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(gt_binary, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("2. Ground Truth Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Probability
    im3 = axes[2].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("3. SegFormer-B0 Probability", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Color-Coded Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("4. Overlay (Green=TP, Red=FP, Blue=FN)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig


def save_sample_visualizations(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    output_dir: str,
    threshold: float = 0.5,
    prefix: str = "sample"
):
    """
    Saves multiple sample visualization figures to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, (sar, gt, pred, name) in enumerate(samples):
        fig = create_prediction_figure(sar, gt, pred, threshold, title=f"Sample: {name}")
        out_path = os.path.join(output_dir, f"{prefix}_{i+1:02d}_{name}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved visualization -> {out_path}")
