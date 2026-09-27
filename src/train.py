"""
Training Engine for OceanTrace SegFormer Experiment 1
-----------------------------------------------------
Full 30-epoch training with:
  - Explicit 3-epoch linear warmup + cosine annealing decay
  - Verbose per-epoch metric printing (Train Loss/IoU/Dice, Val Loss/IoU/Dice/Precision/Recall/F1/TP/FP/FN/TN)
  - Best-model checkpointing (Val IoU) and last-model checkpointing
  - Early stopping (patience=7 on Val IoU)
  - Full experiment metadata + scene split manifest saved to disk
  - Validation sample visualizations saved at best checkpoint
  - GPU enforcement (no silent CPU fallback for full training)
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import yaml
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import build_segformer_b0, verify_model_forward
from src.dataset import get_dataloaders
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetricsMeter


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42, deterministic: bool = True):
    """Sets random seeds for Python, NumPy, PyTorch and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler: Explicit Warmup + Cosine Decay
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 3,
    total_epochs: int = 30,
    min_lr: float = 1e-6,
    base_lr: float = 1e-4
):
    """
    Builds an explicit Linear Warmup + Cosine Annealing schedule:
      Epochs 1..warmup_epochs  : Linear ramp-up from min_lr to base_lr.
      Epochs warmup+1..total   : Cosine decay from base_lr to min_lr.

    NOTE: CosineAnnealingLR alone does NOT perform warmup.
    This function implements warmup explicitly using LinearLR + SequentialLR.
    """
    if warmup_epochs <= 0:
        print("  [Scheduler] Cosine Annealing only (no warmup).")
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    warmup_factor = min_lr / base_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    print(
        f"  [Scheduler] Warmup: {warmup_epochs} epochs "
        f"({min_lr:.1e} → {base_lr:.1e})  |  "
        f"Cosine: {cosine_epochs} epochs ({base_lr:.1e} → {min_lr:.1e})"
    )
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation Steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    model.train()
    meter = SegmentationMetricsMeter(threshold=threshold)
    total_loss = total_bce = total_dice = 0.0
    n = len(loader)

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss, breakdown = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += breakdown["loss"]
        total_bce  += breakdown["bce_loss"]
        total_dice += breakdown["dice_loss"]
        meter.update(logits, masks)

    metrics = meter.compute()
    metrics["loss"]      = total_loss / max(1, n)
    metrics["bce_loss"]  = total_bce  / max(1, n)
    metrics["dice_loss"] = total_dice / max(1, n)
    return metrics["loss"], metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    meter = SegmentationMetricsMeter(threshold=threshold)
    total_loss = total_bce = total_dice = 0.0
    n = len(loader)

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        logits = model(images)
        loss, breakdown = criterion(logits, masks)
        total_loss += breakdown["loss"]
        total_bce  += breakdown["bce_loss"]
        total_dice += breakdown["dice_loss"]
        meter.update(logits, masks)

    metrics = meter.compute()
    metrics["loss"]      = total_loss / max(1, n)
    metrics["bce_loss"]  = total_bce  / max(1, n)
    metrics["dice_loss"] = total_dice / max(1, n)
    return metrics["loss"], metrics


# ─────────────────────────────────────────────────────────────────────────────
# Validation Visualizations
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_val_visualizations(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    fig_dir: str,
    epoch: int,
    threshold: float = 0.5,
    n_samples: int = 5
):
    """Saves up to n_samples 4-panel validation prediction figures."""
    os.makedirs(fig_dir, exist_ok=True)
    model.eval()
    saved = 0

    for batch in val_loader:
        images = batch["image"]      # (B, 1, 256, 256)
        masks  = batch["mask"]       # (B, 1, 256, 256)
        names  = batch["scene_name"]

        logits = model(images.to(device))
        probs  = torch.sigmoid(logits).cpu()

        for i in range(images.shape[0]):
            if saved >= n_samples:
                break

            img_norm  = images[i, 0].numpy()          # (256, 256) in [0,1]
            gt_mask   = masks[i, 0].numpy()            # (256, 256) binary
            pred_prob = probs[i, 0].numpy()            # (256, 256) [0,1]
            pred_bin  = (pred_prob >= threshold).astype(np.float32)
            sname     = names[i] if isinstance(names, (list, tuple)) else "unknown"

            # Reconstruct approximate dB for display (reverse normalize)
            img_db_approx = img_norm * 40.0 - 35.0

            # Overlay
            sar_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
            tp = (pred_bin == 1) & (gt_mask >= 0.5)
            fp = (pred_bin == 1) & (gt_mask <  0.5)
            fn = (pred_bin == 0) & (gt_mask >= 0.5)
            overlay = sar_rgb.copy()
            overlay[tp] = [0.0, 1.0, 0.0]
            overlay[fp] = [1.0, 0.2, 0.2]
            overlay[fn] = [0.2, 0.6, 1.0]

            fig, axes = plt.subplots(1, 4, figsize=(18, 4), dpi=130)
            axes[0].imshow(img_db_approx, cmap="gray", vmin=-35, vmax=5)
            axes[0].set_title("SAR (dB)", fontsize=9, fontweight="bold")
            axes[1].imshow(gt_mask, cmap="viridis", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth", fontsize=9, fontweight="bold")
            im2 = axes[2].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
            axes[2].set_title("Pred Probability", fontsize=9, fontweight="bold")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            axes[3].imshow(np.clip(overlay, 0, 1))
            axes[3].set_title("Overlay (G=TP R=FP B=FN)", fontsize=9, fontweight="bold")
            for ax in axes:
                ax.axis("off")

            oil_gt  = 100.0 * (gt_mask >= 0.5).mean()
            oil_prd = 100.0 * pred_bin.mean()
            fig.suptitle(
                f"Epoch {epoch:02d} | Val Sample {saved+1} | Scene: {sname} | "
                f"GT Oil: {oil_gt:.2f}% | Pred Oil: {oil_prd:.2f}%",
                fontsize=9, fontweight="bold"
            )
            plt.tight_layout()
            fpath = os.path.join(fig_dir, f"val_ep{epoch:02d}_s{saved+1:02d}_{sname.replace('.tif','')}.png")
            plt.savefig(fpath, bbox_inches="tight", dpi=130)
            plt.close(fig)
            saved += 1

        if saved >= n_samples:
            break

    print(f"  [VIZ] Saved {saved} validation visualizations → {fig_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    config_path: str,
    raw_data_dir: Optional[str] = None,
    smoke_test: bool = False,
    allow_cpu: bool = False
) -> Dict[str, Any]:

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ── GPU Enforcement ───────────────────────────────────────────────────────
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not allow_cpu and not smoke_test:
        raise RuntimeError(
            "\n[STOP] CUDA is NOT available!\n"
            "Full Experiment 1 training must run on Google Colab GPU.\n"
            "In Colab: Runtime → Change runtime type → GPU (T4 or A100).\n"
            "Do NOT train on CPU."
        )

    device = torch.device("cuda" if cuda_ok else "cpu")

    print("=" * 58)
    print("  EXPERIMENT 1 — SegFormer-B0 — Sentinel-1 SAR")
    print("  OceanTrace Oil Spill Segmentation Baseline")
    print("=" * 58)
    print("HARDWARE VERIFICATION")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {cuda_ok}")
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Name        : {gpu_name}")
        print(f"  GPU VRAM        : {vram_gb:.2f} GB")
        print(f"  CUDA Version    : {torch.version.cuda}")
        print(f"  Device Count    : {torch.cuda.device_count()}")
        print("  Status          : [PASS] GPU READY")
    else:
        print(f"  Device          : CPU  ({'smoke-test / dev mode' if smoke_test or allow_cpu else 'BLOCKED'})")
    print("=" * 58)

    # ── Reproducibility ────────────────────────────────────────────────────────
    seed = int(cfg["project"]["seed"])
    set_seed(seed, deterministic=cfg["project"].get("deterministic", True))
    print(f"\nReproducibility seed: {seed}  |  deterministic: {cfg['project'].get('deterministic', True)}")

    # ── Directories ───────────────────────────────────────────────────────────
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    log_dir  = cfg["outputs"]["log_dir"]
    fig_dir  = cfg["outputs"]["figure_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(fig_dir,  exist_ok=True)

    # Copy scene split manifests into log_dir for archival
    splits_dir = cfg["data"]["splits_dir"]
    for split_file in ["train_scenes.csv", "val_scenes.csv", "test_scenes.csv"]:
        src = os.path.join(splits_dir, split_file)
        dst = os.path.join(log_dir, split_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir   = raw_data_dir or cfg["data"]["raw_dir"]
    batch_size = int(cfg["training"]["batch_size"])

    print(f"\nDataset directory : {data_dir}")
    print(f"Patch size        : {cfg['data']['patch_size']}×{cfg['data']['patch_size']}")
    print(f"dB clip           : [{cfg['data']['db_clip_min']}, {cfg['data']['db_clip_max']}]")
    print(f"Batch size        : {batch_size}  |  Seed: {seed}")
    print(f"Splits dir        : {splits_dir}\n")

    try:
        train_loader, val_loader = get_dataloaders(
            config=cfg,
            raw_data_dir=data_dir,
            smoke_test=smoke_test,
            batch_size=batch_size
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 8:
            batch_size = int(cfg["training"]["batch_size_fallback"])
            print(f"[OOM] Reducing batch size to {batch_size} and retrying...")
            train_loader, val_loader = get_dataloaders(
                config=cfg,
                raw_data_dir=data_dir,
                smoke_test=smoke_test,
                batch_size=batch_size
            )
        else:
            raise

    print(f"Train patches: {len(train_loader.dataset):,}  →  {len(train_loader)} batches")
    print(f"Val patches  : {len(val_loader.dataset):,}  →  {len(val_loader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    print()
    model = build_segformer_b0(
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        encoder_weights=cfg["model"]["encoder_weights"],
        verbose=True
    ).to(device)

    verify_model_forward(model, device, int(cfg["data"]["patch_size"]))
    print("[PASS] 1-Channel SAR forward pass verified.\n")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = DiceBCELoss(
        dice_weight=float(cfg["loss"]["dice_weight"]),
        bce_weight=float(cfg["loss"]["bce_weight"]),
        smooth=float(cfg["loss"].get("smooth", 1.0))
    )

    base_lr = float(cfg["optimizer"]["lr"])
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg["optimizer"]["weight_decay"])
    )

    epochs        = 1 if smoke_test else int(cfg["training"]["epochs"])
    warmup_epochs = 0 if smoke_test else int(cfg["scheduler"]["warmup_epochs"])
    min_lr        = float(cfg["scheduler"]["min_lr"])

    print("[Scheduler Configuration]")
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=min_lr,
        base_lr=base_lr
    )

    patience  = int(cfg["training"]["early_stopping_patience"])
    threshold = float(cfg["training"]["threshold"])

    best_val_iou   = -1.0
    best_epoch     = -1
    patience_ctr   = 0
    history        = []
    experiment_start = time.time()

    print()
    print("=" * 58)
    mode_str = "SMOKE TEST (1 epoch, minimal data)" if smoke_test else f"FULL TRAINING ({epochs} epochs max)"
    print(f"  STARTING: {mode_str}")
    print(f"  Threshold     : {threshold}")
    print(f"  Early Stop    : patience={patience} (monitor=val_iou)")
    print(f"  Checkpoints   : {ckpt_dir}")
    print("=" * 58)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        t_ep = time.time()

        # Determine schedule phase
        if epoch <= warmup_epochs:
            phase = f"Warmup ({epoch}/{warmup_epochs})"
        else:
            phase = f"Cosine ({epoch - warmup_epochs}/{max(1, epochs - warmup_epochs)})"

        train_loss, tm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, threshold
        )
        val_loss, vm = validate(
            model, val_loader, criterion, device, threshold
        )
        scheduler.step()

        ep_secs = time.time() - t_ep
        val_iou = vm["iou"]

        entry = {
            "epoch":          epoch,
            "phase":          phase,
            "lr":             lr_this_epoch,
            "train_loss":     train_loss,
            "train_iou":      tm["iou"],
            "train_dice":     tm["dice"],
            "train_precision":tm["precision"],
            "train_recall":   tm["recall"],
            "val_loss":       val_loss,
            "val_iou":        vm["iou"],
            "val_dice":       vm["dice"],
            "val_precision":  vm["precision"],
            "val_recall":     vm["recall"],
            "val_f1":         vm["f1"],
            "val_tp":         vm["tp"],
            "val_fp":         vm["fp"],
            "val_fn":         vm["fn"],
            "val_tn":         vm["tn"],
            "time_sec":       ep_secs,
            "patience_ctr":   patience_ctr
        }
        history.append(entry)

        # ── Per-Epoch Verbose Print ───────────────────────────────────────
        print(f"\n{'─'*58}")
        print(f"  Epoch [{epoch:02d}/{epochs:02d}]  Phase: {phase}  Time: {ep_secs:.1f}s")
        print(f"  LR            : {lr_this_epoch:.4e}")
        print(f"  Train Loss    : {train_loss:.5f}  (BCE: {tm['bce_loss']:.5f}  Dice: {tm['dice_loss']:.5f})")
        print(f"  Train IoU     : {tm['iou']:.5f}   Train Dice : {tm['dice']:.5f}")
        print(f"  Val Loss      : {val_loss:.5f}")
        print(f"  Val IoU       : {vm['iou']:.5f}   Val Dice   : {vm['dice']:.5f}")
        print(f"  Val Precision : {vm['precision']:.5f}   Val Recall : {vm['recall']:.5f}   Val F1: {vm['f1']:.5f}")
        print(f"  Val TP:{vm['tp']:,}  FP:{vm['fp']:,}  FN:{vm['fn']:,}  TN:{vm['tn']:,}")

        # ── Checkpointing ─────────────────────────────────────────────────
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            best_epoch   = epoch
            patience_ctr = 0

            best_ckpt = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou":            val_iou,
                "val_dice":           vm["dice"],
                "val_precision":      vm["precision"],
                "val_recall":         vm["recall"],
                "val_f1":             vm["f1"],
                "config":             cfg,
                "seed":               seed,
                "batch_size_used":    batch_size,
                "total_params":       total_params,
                "trainable_params":   trainable_params
            }, best_ckpt)
            print(f"\n  *** NEW BEST — Epoch {epoch:02d} | Val IoU = {val_iou:.5f} | Saved → {best_ckpt}")

            # Save validation visualizations at every new best
            if not smoke_test:
                save_val_visualizations(
                    model, val_loader, device, fig_dir,
                    epoch=epoch, threshold=threshold, n_samples=5
                )
        else:
            patience_ctr += 1
            print(f"  No improvement — patience {patience_ctr}/{patience}")

        # Always save last checkpoint
        last_ckpt = os.path.join(ckpt_dir, "last_model.pth")
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou":            val_iou,
            "config":             cfg,
            "seed":               seed
        }, last_ckpt)

        # ── Early Stopping ─────────────────────────────────────────────────
        if not smoke_test and patience_ctr >= patience:
            print(f"\n{'='*58}")
            print(f"  [EARLY STOPPING] No Val IoU improvement for {patience} epochs.")
            print(f"  Stopped at epoch {epoch}. Best was epoch {best_epoch} (IoU={best_val_iou:.5f}).")
            print(f"{'='*58}")
            break

    # ── Training Complete ──────────────────────────────────────────────────────
    total_secs = time.time() - experiment_start
    total_min  = total_secs / 60.0
    epochs_run = len(history)

    print(f"\n{'='*58}")
    print("  TRAINING COMPLETE")
    print(f"  Epochs run      : {epochs_run}/{epochs}")
    print(f"  Total time      : {total_min:.1f} min ({total_secs:.0f}s)")
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Best Val IoU    : {best_val_iou:.5f}")
    print(f"{'='*58}")

    # ── Save Full History ──────────────────────────────────────────────────────
    history_df  = pd.DataFrame(history)
    history_csv = os.path.join(log_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    metadata = {
        "experiment":         "OceanTrace_SegFormer_Exp1",
        "model":              "SegFormer-B0 (mit_b0)",
        "in_channels":        cfg["model"]["in_channels"],
        "classes":            cfg["model"]["classes"],
        "encoder_weights":    cfg["model"]["encoder_weights"],
        "total_params":       total_params,
        "trainable_params":   trainable_params,
        "seed":               seed,
        "epochs_run":         epochs_run,
        "epochs_max":         epochs,
        "batch_size_used":    batch_size,
        "warmup_epochs":      warmup_epochs,
        "base_lr":            base_lr,
        "min_lr":             min_lr,
        "weight_decay":       cfg["optimizer"]["weight_decay"],
        "loss":               "0.5*Dice + 0.5*BCE",
        "augmentation":       "HorizontalFlip, VerticalFlip, RandomRotate90",
        "patch_size":         cfg["data"]["patch_size"],
        "db_clip":            [cfg["data"]["db_clip_min"], cfg["data"]["db_clip_max"]],
        "threshold":          threshold,
        "early_stopping_patience": patience,
        "best_epoch":         best_epoch,
        "best_val_iou":       best_val_iou,
        "total_train_time_min": total_min,
        "device":             str(device),
        "gpu_name":           torch.cuda.get_device_name(0) if cuda_ok else "CPU",
        "pytorch_version":    torch.__version__,
        "cuda_version":       torch.version.cuda if cuda_ok else None,
        "data_dir":           data_dir,
        "splits_dir":         splits_dir,
        "history": history
    }

    metadata_path = os.path.join(log_dir, "experiment1_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nLogs saved:")
    print(f"  Training history CSV  : {history_csv}")
    print(f"  Experiment metadata   : {metadata_path}")

    return {
        "status":           "SUCCESS",
        "best_val_iou":     best_val_iou,
        "best_epoch":       best_epoch,
        "epochs_run":       epochs_run,
        "total_time_min":   total_min,
        "history":          history,
        "best_checkpoint":  os.path.join(ckpt_dir, "best_model.pth"),
        "last_checkpoint":  os.path.join(ckpt_dir, "last_model.pth"),
        "metadata_path":    metadata_path,
        "history_csv":      history_csv
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer-B0 — OceanTrace Experiment 1")
    parser.add_argument("--config",     type=str, default="configs/exp1_segformer_b0.yaml")
    parser.add_argument("--data_dir",   type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--allow_cpu",  action="store_true")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        raw_data_dir=args.data_dir,
        smoke_test=args.smoke_test,
        allow_cpu=args.allow_cpu
    )
