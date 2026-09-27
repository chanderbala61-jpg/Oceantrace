"""
Pre-Training Validation Check for OceanTrace SegFormer Experiment 1
--------------------------------------------------------------------
Loads the smoke-test checkpoint, runs inference on VALIDATION scenes only,
and performs a thorough diagnostic report before full training is approved.

Usage on Colab:
    python scripts/pre_training_validation.py \
        --data_dir /content/data/raw \
        --checkpoint outputs/checkpoints/best_model.pth \
        --n_samples 5

Checks performed:
    1. Checkpoint load integrity
    2. Tensor shape consistency (image and mask)
    3. Channel count correctness (must be 1-channel SAR)
    4. Normalization range verification ([0.0, 1.0])
    5. Mask value uniqueness (binary: 0.0 and/or 1.0 only)
    6. Inference output shape match
    7. Prediction not all-zero / all-one
    8. Sigmoid probability range
    9. Spatial alignment (image vs mask)
    10. Oil pixel fraction sanity (prediction vs ground truth)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import tifffile
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab script mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_segformer_b0
from src.transforms import preprocess_sar_db


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Utilities
# ─────────────────────────────────────────────────────────────────────────────

class CheckReport:
    """Accumulates PASS/FAIL check results and prints a final summary."""

    def __init__(self):
        self.results = []

    def record(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({"check": name, "status": status, "detail": detail})
        icon = "✓" if passed else "✗"
        print(f"  [{status}] {icon} {name}" + (f": {detail}" if detail else ""))
        return passed

    def summary(self):
        failed = [r for r in self.results if r["status"] == "FAIL"]
        total = len(self.results)
        print("\n" + "=" * 52)
        print(f"VALIDATION SUMMARY: {total - len(failed)}/{total} checks PASSED")
        print("=" * 52)
        if failed:
            print("FAILED CHECKS:")
            for r in failed:
                print(f"  ✗ {r['check']}: {r['detail']}")
        else:
            print("ALL CHECKS PASSED — pipeline is ready for full 30-epoch training.")
        print("=" * 52)
        return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Patch Sampler from Validation Scenes
# ─────────────────────────────────────────────────────────────────────────────

def sample_patches_from_val_scenes(
    val_split_csv: str,
    raw_data_dir: str,
    n_samples: int = 5,
    patch_size: int = 256,
    db_min: float = -35.0,
    db_max: float = 5.0,
    seed: int = 42
):
    """
    Loads validation scenes and samples patches that contain oil spills.
    Returns raw SAR patches (dB), normalized patches, binary masks, and scene names.
    """
    df_val = pd.read_csv(val_split_csv)
    rng = np.random.RandomState(seed)

    all_samples = []

    for _, row in df_val.iterrows():
        img_path = os.path.join(raw_data_dir, row["relative_img_path"])
        mask_path = os.path.join(raw_data_dir, row["relative_mask_path"])
        scene_name = row["scene_name"]

        if not os.path.exists(img_path):
            print(f"  [WARN] Missing image: {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  [WARN] Missing mask: {mask_path}")
            continue

        img = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.float32)

        # Ensure 2D
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        h, w = img.shape

        # Find positive patch locations (oil spill present)
        stride = 256
        candidate_positions = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                m_patch = mask[y: y + patch_size, x: x + patch_size]
                if np.sum(m_patch > 0.5) >= 50:  # At least 50 oil pixels
                    candidate_positions.append((y, x))

        if len(candidate_positions) == 0:
            # Fall back to any patch
            if h >= patch_size and w >= patch_size:
                y = rng.randint(0, max(1, h - patch_size))
                x = rng.randint(0, max(1, w - patch_size))
                candidate_positions = [(y, x)]
            else:
                continue

        # Sample up to 2 patches per scene
        rng.shuffle(candidate_positions)
        chosen = candidate_positions[:2]

        for (y, x) in chosen:
            img_patch_db = img[y: y + patch_size, x: x + patch_size]
            mask_patch = mask[y: y + patch_size, x: x + patch_size]

            # Clean NaN/Inf
            img_patch_db = np.nan_to_num(img_patch_db, nan=-35.0, posinf=5.0, neginf=-35.0)
            mask_patch = np.nan_to_num(mask_patch, nan=0.0)

            img_patch_norm = preprocess_sar_db(img_patch_db, db_min, db_max)

            all_samples.append({
                "scene_name": scene_name,
                "coords": (y, x),
                "img_db": img_patch_db,         # float32 dB (256,256) — for visualization
                "img_norm": img_patch_norm,      # float32 [0,1] (256,256) — model input
                "mask": mask_patch               # float32 binary (256,256)
            })

        if len(all_samples) >= n_samples * 2:
            break

    # Trim to requested number
    rng.shuffle(all_samples)
    return all_samples[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_validation_figure(
    sar_db: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_binary: np.ndarray,
    scene_name: str,
    sample_idx: int,
    output_path: str,
    threshold: float = 0.5
):
    """Saves a 5-panel validation diagnostic figure."""
    fig = plt.figure(figsize=(22, 5), dpi=150)
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    # Build overlay: Green=TP, Red=FP, Cyan=FN
    h, w = sar_db.shape
    sar_norm_vis = (np.clip(sar_db, -35.0, 5.0) - (-35.0)) / 40.0
    overlay = np.stack([sar_norm_vis, sar_norm_vis, sar_norm_vis], axis=-1)

    tp = (pred_binary == 1) & (gt_mask >= 0.5)
    fp = (pred_binary == 1) & (gt_mask < 0.5)
    fn = (pred_binary == 0) & (gt_mask >= 0.5)

    overlay[tp] = [0.0, 1.0, 0.0]   # Green: correct detection
    overlay[fp] = [1.0, 0.2, 0.2]   # Red: false alarm
    overlay[fn] = [0.2, 0.6, 1.0]   # Cyan: missed oil

    titles = [
        f"1. SAR (dB)\n[{sar_db.min():.1f}, {sar_db.max():.1f}]",
        "2. Ground Truth",
        "3. Prediction Prob.",
        "4. Prediction Binary\n(threshold=0.5)",
        "5. Overlay\nGreen=TP Red=FP Blue=FN"
    ]
    data = [
        (sar_db,      "gray",    -35.0, 5.0),
        (gt_mask,     "viridis", 0.0,   1.0),
        (pred_prob,   "magma",   0.0,   1.0),
        (pred_binary, "viridis", 0.0,   1.0),
        (overlay,     None,      None,  None)
    ]

    for i, (ax_data, cmap, vmin, vmax) in enumerate(data):
        ax = fig.add_subplot(gs[i])
        if cmap is None:
            ax.imshow(np.clip(ax_data, 0, 1))
        else:
            ax.imshow(ax_data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=9, fontweight="bold")
        ax.axis("off")

    # Stats annotation
    oil_pred_pct = 100.0 * pred_binary.mean()
    oil_gt_pct = 100.0 * (gt_mask >= 0.5).mean()
    fig.suptitle(
        f"Validation Sample {sample_idx} | Scene: {scene_name} | "
        f"GT Oil: {oil_gt_pct:.2f}% | Pred Oil: {oil_pred_pct:.2f}%",
        fontsize=10, fontweight="bold", y=1.01
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Validation Routine
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_training_validation(
    config_path: str,
    checkpoint_path: str,
    raw_data_dir: str = None,
    n_samples: int = 5,
    allow_cpu: bool = False
):
    report = CheckReport()
    saved_figures = []

    print("=" * 52)
    print("PRE-TRAINING VALIDATION CHECK")
    print("OceanTrace SegFormer Experiment 1")
    print("=" * 52)

    # ── Load Config ──────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir  = raw_data_dir or cfg["data"]["raw_dir"]
    splits_dir = cfg["data"]["splits_dir"]
    db_min    = cfg["data"]["db_clip_min"]
    db_max    = cfg["data"]["db_clip_max"]
    patch_size = cfg["data"]["patch_size"]
    threshold  = cfg["training"]["threshold"]
    seed       = cfg["project"]["seed"]
    fig_dir    = cfg["outputs"]["figure_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    val_split_csv = os.path.join(splits_dir, "val_scenes.csv")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if not torch.cuda.is_available() and not allow_cpu:
        print("[WARN] No CUDA GPU. Using CPU for validation diagnostics only.")

    # ── Step 1: Load Checkpoint ───────────────────────────────────────────────
    print("\n─── STEP 1: Checkpoint Load ────────────────────────────────")
    ckpt_exists = os.path.exists(checkpoint_path)
    report.record("Checkpoint file exists", ckpt_exists, checkpoint_path)
    if not ckpt_exists:
        print("[FATAL] Checkpoint not found. Cannot continue.")
        report.summary()
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_keys_ok = "model_state_dict" in ckpt and "epoch" in ckpt
    report.record("Checkpoint has required keys", ckpt_keys_ok,
                  f"keys={list(ckpt.keys())}")
    saved_epoch = ckpt.get("epoch", "?")
    saved_iou   = ckpt.get("val_iou", "?")
    print(f"    Smoke-test checkpoint: epoch={saved_epoch}, val_iou={saved_iou}")

    # ── Step 2: Model Construction ───────────────────────────────────────────
    print("\n─── STEP 2: Model Construction ─────────────────────────────")
    model = build_segformer_b0(
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        encoder_weights=None,
        verbose=True
    ).to(device)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        report.record("Weights loaded into model", True)
    except Exception as e:
        report.record("Weights loaded into model", False, str(e))
        report.summary()
        return

    # ── Step 3: Sample Validation Patches ────────────────────────────────────
    print("\n─── STEP 3: Sampling Validation Patches ────────────────────")
    val_split_exists = os.path.exists(val_split_csv)
    report.record("Val split CSV exists", val_split_exists, val_split_csv)
    if not val_split_exists:
        report.summary()
        return

    samples = sample_patches_from_val_scenes(
        val_split_csv=val_split_csv,
        raw_data_dir=data_dir,
        n_samples=n_samples,
        patch_size=patch_size,
        db_min=db_min,
        db_max=db_max,
        seed=seed
    )
    report.record("Enough val samples loaded", len(samples) >= 1,
                  f"loaded {len(samples)} samples")
    if len(samples) == 0:
        report.summary()
        return

    # ── Per-Sample Inference Loop ─────────────────────────────────────────────
    print("\n─── STEP 4: Per-Sample Diagnostic Inference ─────────────────")
    all_results = []

    for idx, sample in enumerate(samples):
        sname  = sample["scene_name"]
        img_db = sample["img_db"]       # (256,256) float32 dB
        img_nrm = sample["img_norm"]    # (256,256) float32 [0,1]
        gt_mask = sample["mask"]        # (256,256) float32 binary

        print(f"\n  Sample {idx+1}/{len(samples)} | Scene: {sname}")

        # ── Tensor Shapes ─────────────────────────────────────────────────
        img_tensor = torch.from_numpy(img_nrm).unsqueeze(0).unsqueeze(0).float().to(device)
        # Shape: (1, 1, 256, 256)

        img_shape_ok = img_tensor.shape == (1, 1, patch_size, patch_size)
        report.record(f"S{idx+1}: Image tensor shape (1,1,256,256)",
                      img_shape_ok, str(tuple(img_tensor.shape)))

        ch_ok = img_tensor.shape[1] == 1
        report.record(f"S{idx+1}: Input channel count == 1",
                      ch_ok, f"channels={img_tensor.shape[1]}")

        # ── Normalization Range ────────────────────────────────────────────
        img_min = img_tensor.min().item()
        img_max = img_tensor.max().item()
        norm_ok = (img_min >= -0.01) and (img_max <= 1.01)
        report.record(f"S{idx+1}: Normalized input in [0,1]",
                      norm_ok, f"range=[{img_min:.4f}, {img_max:.4f}]")

        # ── GT Mask Values ─────────────────────────────────────────────────
        gt_unique = np.unique(gt_mask)
        gt_binary_ok = all(v in [0.0, 1.0] for v in gt_unique)
        report.record(f"S{idx+1}: GT mask strictly binary (0/1)",
                      gt_binary_ok, f"unique={gt_unique}")

        # ── Run Inference ─────────────────────────────────────────────────
        with torch.no_grad():
            logits = model(img_tensor)
            probs  = torch.sigmoid(logits)

        # ── Output Shape Check ─────────────────────────────────────────────
        out_shape_ok = logits.shape == (1, 1, patch_size, patch_size)
        report.record(f"S{idx+1}: Output logit shape (1,1,256,256)",
                      out_shape_ok, str(tuple(logits.shape)))

        # ── Prob Range Check ──────────────────────────────────────────────
        p_min = probs.min().item()
        p_max = probs.max().item()
        prob_range_ok = (p_min >= 0.0) and (p_max <= 1.0)
        report.record(f"S{idx+1}: Sigmoid probs in [0.0, 1.0]",
                      prob_range_ok, f"range=[{p_min:.4f}, {p_max:.4f}]")

        # ── Convert to Numpy ──────────────────────────────────────────────
        pred_prob   = probs.squeeze().cpu().numpy()      # (256,256) float32
        pred_binary = (pred_prob >= threshold).astype(np.float32)  # (256,256)

        pred_unique = np.unique(pred_binary)
        pred_binary_ok = all(v in [0.0, 1.0] for v in pred_unique)
        report.record(f"S{idx+1}: Pred binary contains valid values (0/1)",
                      pred_binary_ok, f"unique={pred_unique}")

        # ── Not All-Zero / All-One ─────────────────────────────────────────
        not_all_zero = pred_binary.sum() > 0
        not_all_one  = pred_binary.mean() < 0.99
        report.record(f"S{idx+1}: Prediction not all-zero",
                      not_all_zero, f"predicted_oil={pred_binary.mean()*100:.3f}%")
        report.record(f"S{idx+1}: Prediction not all-one",
                      not_all_one,  f"predicted_oil={pred_binary.mean()*100:.3f}%")

        # ── Spatial Alignment: image and mask same shape ───────────────────
        spatial_ok = (img_db.shape == gt_mask.shape == pred_prob.shape)
        report.record(f"S{idx+1}: Image/mask/pred spatially aligned",
                      spatial_ok,
                      f"img={img_db.shape}, mask={gt_mask.shape}, pred={pred_prob.shape}")

        # ── Oil Pixel Fractions ────────────────────────────────────────────
        pct_pred_oil = 100.0 * pred_binary.mean()
        pct_gt_oil   = 100.0 * (gt_mask >= 0.5).mean()

        # Print detailed sample report
        print(f"    Image tensor shape:          {tuple(img_tensor.shape)}")
        print(f"    GT mask shape:               {gt_mask.shape}")
        print(f"    Logit output shape:          {tuple(logits.shape)}")
        print(f"    Pred prob shape:             {pred_prob.shape}")
        print(f"    Pred binary unique values:   {pred_unique}")
        print(f"    GT mask unique values:       {gt_unique}")
        print(f"    Normalized input range:      [{img_min:.4f}, {img_max:.4f}]")
        print(f"    Sigmoid prob range:          [{p_min:.4f}, {p_max:.4f}]")
        print(f"    GT oil pixels:               {pct_gt_oil:.3f}%")
        print(f"    Predicted oil pixels:        {pct_pred_oil:.3f}%")

        # ── Mask Inversion Check ────────────────────────────────────────────
        # If model predicts much more oil than GT, check for possible inversion
        inversion_suspected = pct_gt_oil < 5.0 and pct_pred_oil > 50.0
        report.record(f"S{idx+1}: No mask inversion suspected",
                      not inversion_suspected,
                      f"gt_oil={pct_gt_oil:.2f}%, pred_oil={pct_pred_oil:.2f}%")

        all_results.append({
            "sample_idx":     idx + 1,
            "scene_name":     sname,
            "pct_gt_oil":     pct_gt_oil,
            "pct_pred_oil":   pct_pred_oil,
            "pred_unique":    pred_unique.tolist(),
            "gt_unique":      gt_unique.tolist(),
            "img_shape":      tuple(img_tensor.shape),
            "pred_shape":     tuple(logits.shape),
            "prob_range":     [p_min, p_max]
        })

        # ── Save Visualization ─────────────────────────────────────────────
        fig_name = f"val_check_sample_{idx+1:02d}_{sname.replace('.tif','')}.png"
        fig_path = os.path.join(fig_dir, fig_name)
        save_validation_figure(
            sar_db=img_db,
            gt_mask=gt_mask,
            pred_prob=pred_prob,
            pred_binary=pred_binary,
            scene_name=sname,
            sample_idx=idx + 1,
            output_path=fig_path,
            threshold=threshold
        )
        saved_figures.append(fig_path)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n─── STEP 5: Saved Visualizations ────────────────────────────")
    for fp in saved_figures:
        print(f"  → {fp}")

    print("\n─── STEP 6: Per-Sample Statistics ───────────────────────────")
    df_results = pd.DataFrame(all_results)
    print(df_results[[
        "sample_idx", "scene_name", "pct_gt_oil", "pct_pred_oil",
        "gt_unique", "pred_unique"
    ]].to_string(index=False))

    all_ok = report.summary()

    print("\n─── FINAL READINESS VERDICT ─────────────────────────────────")
    if all_ok:
        print("✓ PIPELINE IS READY FOR FULL 30-EPOCH TRAINING.")
        print("  Run: python scripts/colab_train.py --train --data_dir <DATA_DIR>")
    else:
        print("✗ ISSUES DETECTED. Fix the above FAIL items before starting full training.")

    return {
        "all_passed": all_ok,
        "saved_figures": saved_figures,
        "sample_results": all_results,
        "report": report.results
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-Training Validation Check for OceanTrace SegFormer Exp 1"
    )
    parser.add_argument("--config",     type=str,
                        default="configs/exp1_segformer_b0.yaml",
                        help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/best_model.pth",
                        help="Path to smoke-test checkpoint")
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Override raw dataset directory")
    parser.add_argument("--n_samples",  type=int, default=5,
                        help="Number of validation patches to visualize")
    parser.add_argument("--allow_cpu",  action="store_true",
                        help="Allow CPU execution for local testing")
    args = parser.parse_args()

    run_pre_training_validation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        raw_data_dir=args.data_dir,
        n_samples=args.n_samples,
        allow_cpu=args.allow_cpu
    )
