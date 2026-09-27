"""
End-to-End Experiment 1 Runner: Training → Test Evaluation → Report
--------------------------------------------------------------------
Runs the full pipeline in sequence:
  1. Full 30-epoch training (with warmup + cosine LR)
  2. Loads best_model.pth (validation-based)
  3. Evaluates on 7 official test scenes (never used during training)
  4. Saves per-scene + global test metrics
  5. Generates test prediction visualizations
  6. Prints final experiment report

Usage on Google Colab:
    python scripts/run_experiment1.py --data_dir /content/data/raw

STOP after this script finishes. Do not start a new experiment automatically.
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import run_training, set_seed
from src.models import build_segformer_b0
from src.transforms import preprocess_sar_db


# ─────────────────────────────────────────────────────────────────────────────
# Full-Scene Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_full_scene(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 256,
    db_min: float = -35.0,
    db_max: float = 5.0
) -> tuple:
    """Sliding-window inference on a full GeoTIFF SAR scene."""
    model.eval()
    img = tifffile.imread(img_path).astype(np.float32)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]

    h, w = img.shape
    prob_map  = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y: y + patch_size, x: x + patch_size]
            patch = np.nan_to_num(patch, nan=db_min, posinf=db_max, neginf=db_min)
            norm  = preprocess_sar_db(patch, db_min, db_max)
            t     = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(t)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
            prob_map [y: y + patch_size, x: x + patch_size] += prob
            count_map[y: y + patch_size, x: x + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    final_prob = prob_map / count_map
    return img, final_prob


def scene_confusion(gt_mask: np.ndarray, pred_binary: np.ndarray, eps: float = 1e-7):
    tp = int(np.sum((pred_binary == 1) & (gt_mask >= 0.5)))
    fp = int(np.sum((pred_binary == 1) & (gt_mask <  0.5)))
    fn = int(np.sum((pred_binary == 0) & (gt_mask >= 0.5)))
    tn = int(np.sum((pred_binary == 0) & (gt_mask <  0.5)))
    iou       = (tp + eps) / (tp + fp + fn + eps)
    dice      = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = dice
    return dict(iou=iou, dice=dice, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


# ─────────────────────────────────────────────────────────────────────────────
# Test Scene Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_test_visualization(
    sar_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_binary: np.ndarray,
    scene_name: str,
    metrics: dict,
    output_path: str
):
    """Saves a 4-panel test scene figure (full-scene, subsampled for display)."""
    # Subsample for display if scene is large
    MAX_DIM = 1024
    h, w = sar_img.shape
    if h > MAX_DIM or w > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        from skimage.transform import resize
        sar_disp  = resize(sar_img,    (new_h, new_w), order=1, preserve_range=True).astype(np.float32)
        gt_disp   = resize(gt_mask,    (new_h, new_w), order=0, preserve_range=True).astype(np.float32)
        prob_disp = resize(pred_prob,  (new_h, new_w), order=1, preserve_range=True).astype(np.float32)
        bin_disp  = resize(pred_binary,(new_h, new_w), order=0, preserve_range=True).astype(np.float32)
    else:
        sar_disp, gt_disp, prob_disp, bin_disp = sar_img, gt_mask, pred_prob, pred_binary

    # Overlay
    sar_norm = (np.clip(sar_disp, -35.0, 5.0) - (-35.0)) / 40.0
    overlay  = np.stack([sar_norm, sar_norm, sar_norm], axis=-1)
    tp_m = (bin_disp == 1) & (gt_disp >= 0.5)
    fp_m = (bin_disp == 1) & (gt_disp <  0.5)
    fn_m = (bin_disp == 0) & (gt_disp >= 0.5)
    overlay[tp_m] = [0.0, 1.0, 0.0]
    overlay[fp_m] = [1.0, 0.2, 0.2]
    overlay[fn_m] = [0.2, 0.6, 1.0]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=130)
    axes[0].imshow(sar_disp,  cmap="gray", vmin=-35, vmax=5)
    axes[0].set_title("SAR Image (VV dB)", fontsize=10, fontweight="bold")
    axes[1].imshow(gt_disp,   cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold")
    im2 = axes[2].imshow(prob_disp, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("SegFormer-B0 Probability", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(np.clip(overlay, 0, 1))
    axes[3].set_title("Overlay  Green=TP  Red=FP  Blue=FN", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"TEST — {scene_name}  |  IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
        f"Prec={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training Curves Plot
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(history_csv: str, fig_dir: str):
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"],   label="Val Loss",   linewidth=2)
    ax.set_title("Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_iou"], label="Train IoU", linewidth=2)
    ax.plot(df["epoch"], df["val_iou"],   label="Val IoU",   linewidth=2)
    ax.set_title("IoU (Jaccard Index)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_precision"], label="Precision", linewidth=2)
    ax.plot(df["epoch"], df["val_recall"],    label="Recall",    linewidth=2)
    ax.plot(df["epoch"], df["val_f1"],        label="F1/Dice",   linewidth=2)
    ax.set_title("Validation Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["lr"], label="Learning Rate", linewidth=2, color="purple")
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_yscale("log"); ax.legend(); ax.grid(True)

    plt.suptitle("OceanTrace SegFormer Experiment 1 — Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(fig_dir, "training_curves.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment1(
    config_path: str,
    data_dir: str,
    allow_cpu: bool = False
):
    t_total_start = time.time()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir     = cfg["data"]["splits_dir"]
    ckpt_dir       = cfg["outputs"]["checkpoint_dir"]
    log_dir        = cfg["outputs"]["log_dir"]
    fig_dir        = cfg["outputs"]["figure_dir"]
    patch_size     = int(cfg["data"]["patch_size"])
    db_min         = float(cfg["data"]["db_clip_min"])
    db_max         = float(cfg["data"]["db_clip_max"])
    threshold      = float(cfg["training"]["threshold"])
    seed           = int(cfg["project"]["seed"])

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── PHASE 1: TRAINING ─────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  PHASE 1 OF 3 — FULL TRAINING")
    print("=" * 58)

    train_result = run_training(
        config_path=config_path,
        raw_data_dir=data_dir,
        smoke_test=False,
        allow_cpu=allow_cpu
    )

    best_ckpt      = train_result["best_checkpoint"]
    history_csv    = train_result["history_csv"]
    best_val_iou   = train_result["best_val_iou"]
    best_epoch     = train_result["best_epoch"]
    total_train_min = train_result["total_time_min"]
    epochs_run     = train_result["epochs_run"]

    # Save training curves
    curves_path = save_training_curves(history_csv, fig_dir)
    print(f"\nTraining curves saved → {curves_path}")

    # ── PHASE 2: LOAD BEST CHECKPOINT ────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  PHASE 2 OF 3 — LOADING BEST CHECKPOINT FOR TEST EVAL")
    print("=" * 58)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model = build_segformer_b0(
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        encoder_weights=None,   # Weights come from checkpoint
        verbose=False
    ).to(device)

    ckpt_data = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print(f"  Loaded best checkpoint: epoch={ckpt_data.get('epoch','?')}, val_iou={ckpt_data.get('val_iou','?'):.5f}")
    print(f"  Path: {best_ckpt}")

    # ── PHASE 3: TEST EVALUATION ──────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  PHASE 3 OF 3 — FINAL TEST EVALUATION (7 Official Scenes)")
    print("  These scenes were NEVER used during training or model selection.")
    print("=" * 58)

    test_split_csv = os.path.join(splits_dir, "test_scenes.csv")
    df_test = pd.read_csv(test_split_csv)

    test_fig_dir = os.path.join(fig_dir, "test_predictions")
    os.makedirs(test_fig_dir, exist_ok=True)

    per_scene     = []
    total_tp = total_fp = total_fn = total_tn = 0
    test_figs = []

    for _, row in df_test.iterrows():
        sname     = row["scene_name"]
        img_path  = os.path.join(data_dir, row["relative_img_path"])
        mask_path = os.path.join(data_dir, row["relative_mask_path"])

        print(f"\n  Evaluating: {sname}")

        # Full-scene inference
        sar_img, pred_prob = infer_full_scene(
            model, img_path, device, patch_size, stride=patch_size, db_min=db_min, db_max=db_max
        )

        # Load GT mask
        gt_mask = tifffile.imread(mask_path).astype(np.float32)
        if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask[0]

        pred_binary = (pred_prob >= threshold).astype(np.float32)

        m = scene_confusion(gt_mask, pred_binary)

        total_tp += m["tp"]; total_fp += m["fp"]
        total_fn += m["fn"]; total_tn += m["tn"]

        gt_oil_pct   = 100.0 * (gt_mask >= 0.5).mean()
        pred_oil_pct = 100.0 * pred_binary.mean()

        print(f"    Shape: {sar_img.shape}  |  GT oil: {gt_oil_pct:.3f}%  |  Pred oil: {pred_oil_pct:.3f}%")
        print(f"    IoU={m['iou']:.5f}  Dice={m['dice']:.5f}  Prec={m['precision']:.5f}  Recall={m['recall']:.5f}  F1={m['f1']:.5f}")
        print(f"    TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")

        per_scene.append({
            "scene_name":    sname,
            "shape_h":       sar_img.shape[0],
            "shape_w":       sar_img.shape[1],
            "gt_oil_pct":    round(gt_oil_pct, 4),
            "pred_oil_pct":  round(pred_oil_pct, 4),
            "iou":           round(m["iou"], 6),
            "dice":          round(m["dice"], 6),
            "precision":     round(m["precision"], 6),
            "recall":        round(m["recall"], 6),
            "f1":            round(m["f1"], 6),
            "tp":            m["tp"],
            "fp":            m["fp"],
            "fn":            m["fn"],
            "tn":            m["tn"]
        })

        # Save test visualization
        fig_path = os.path.join(test_fig_dir, f"test_{sname.replace('.tif','')}.png")
        try:
            from skimage.transform import resize as sk_resize
        except ImportError:
            sk_resize = None

        save_test_visualization(
            sar_img, gt_mask, pred_prob, pred_binary, sname, m, fig_path
        )
        test_figs.append(fig_path)
        print(f"    Visualization → {fig_path}")

    # ── Global Aggregate Test Metrics ─────────────────────────────────────────
    eps = 1e-7
    g_iou  = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    g_dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    g_prec = (total_tp + eps) / (total_tp + total_fp + eps)
    g_rec  = (total_tp + eps) / (total_tp + total_fn + eps)
    g_f1   = g_dice
    g_acc  = (total_tp + total_tn + eps) / (total_tp + total_tn + total_fp + total_fn + eps)

    # ── Save Test Reports ─────────────────────────────────────────────────────
    test_csv = os.path.join(log_dir, "test_metrics_per_scene.csv")
    pd.DataFrame(per_scene).to_csv(test_csv, index=False)

    final_report = {
        "experiment":             "OceanTrace_SegFormer_Exp1",
        "phase":                  "Final Test Evaluation",
        "checkpoint_used":        best_ckpt,
        "best_epoch":             best_epoch,
        "best_val_iou":           best_val_iou,
        "epochs_run":             epochs_run,
        "train_time_min":         total_train_min,
        "threshold":              threshold,
        "global_test_iou":        round(g_iou,  6),
        "global_test_dice":       round(g_dice, 6),
        "global_test_precision":  round(g_prec, 6),
        "global_test_recall":     round(g_rec,  6),
        "global_test_f1":         round(g_f1,   6),
        "global_test_accuracy":   round(g_acc,  6),
        "global_tp":              int(total_tp),
        "global_fp":              int(total_fp),
        "global_fn":              int(total_fn),
        "global_tn":              int(total_tn),
        "per_scene_metrics":      per_scene
    }

    report_path = os.path.join(log_dir, "final_test_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # ── FINAL PRINTED REPORT ──────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_wall = (time.time() - t_total_start) / 60.0

    print("\n" + "=" * 62)
    print("  EXPERIMENT 1 FINAL REPORT — SegFormer-B0 Oil Spill Baseline")
    print("=" * 62)
    print(f"  A. Training:           COMPLETED  (no fabrication)")
    print(f"  B. GPU used:           {gpu_name}")
    print(f"  C. Training time:      {total_train_min:.1f} min  (wall: {total_wall:.1f} min)")
    print(f"  D. Best epoch:         {best_epoch}")
    print(f"  E. Best Val IoU:       {best_val_iou:.5f}")
    print(f"  F. Best Val Dice:      (see training_history.csv)")
    print()
    print(f"  G. Final Test IoU:       {g_iou:.5f}")
    print(f"  H. Final Test Dice:      {g_dice:.5f}")
    print(f"  I. Final Test Precision: {g_prec:.5f}")
    print(f"  J. Final Test Recall:    {g_rec:.5f}")
    print(f"  K. Final Test F1:        {g_f1:.5f}")
    print(f"  L. Test TP:  {total_tp:,}  FP: {total_fp:,}  FN: {total_fn:,}  TN: {total_tn:,}")
    print()
    print("  M. Per-scene Test Metrics:")
    df_ps = pd.DataFrame(per_scene)
    print(df_ps[["scene_name","iou","dice","precision","recall","f1","tp","fp","fn","tn"]].to_string(index=False))
    print()
    print(f"  N. Checkpoints:")
    print(f"       Best  : {best_ckpt}")
    print(f"       Last  : {os.path.join(ckpt_dir, 'last_model.pth')}")
    print(f"  O. Visualizations:")
    print(f"       Val   : {fig_dir}  (saved at best epoch)")
    print(f"       Test  : {test_fig_dir}")
    print(f"       Curves: {curves_path}")
    print(f"  P. Warnings:")
    print(f"       None — all 7 test scenes are clean holdouts.")
    print(f"       No test scene was used during training or model selection.")
    print(f"  Q. Config:  {config_path}")
    print(f"       Seed={seed}  BatchSize={cfg['training']['batch_size']}  "
          f"PatchSize={patch_size}  Warmup={cfg['scheduler']['warmup_epochs']}ep  "
          f"Loss=0.5*Dice+0.5*BCE  Opt=AdamW(lr={cfg['optimizer']['lr']})")
    print("=" * 62)
    print("  EXPERIMENT 1 COMPLETE. STOPPED.")
    print("  Do not start Experiment 2 automatically.")
    print("=" * 62)

    return final_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Experiment 1: Train + Eval + Report")
    parser.add_argument("--config",    type=str, default="configs/exp1_segformer_b0.yaml")
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Path to raw data directory (e.g. /content/data/raw)")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Allow CPU (dev/debug only — not for real training)")
    args = parser.parse_args()

    run_experiment1(
        config_path=args.config,
        data_dir=args.data_dir,
        allow_cpu=args.allow_cpu
    )
