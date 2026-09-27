"""
OceanTrace Experiment 2: Local CPU Resumable Chunk-Training Engine (Optimized)
-----------------------------------------------------------------------------
Trains SegFormer-B0 on Sentinel-1 dual-polarization SAR imagery using
sequential, memory-safe dataset chunks on local CPU hardware.

Key Features & Enhancements:
1. Two-Tier Validation Schedule:
   - Chunk 1: FULL validation (180 scenes / 360 patches) - Completed.
   - Chunks 2-9, 11-19, 21-29, 31-41: QUICK validation (fixed 20 scenes / 40 patches, ~1 min).
   - Chunks 10, 20, 30, 42: FULL validation (180 scenes / 360 patches).
2. Scientific Rigor for Best Model:
   - 'best.pth' is strictly governed ONLY by FULL validation IoU.
   - Quick validation is purely diagnostic and does not overwrite best.pth.
3. Checkpoint Integrity:
   - Preserves model_state_dict, optimizer_state_dict, scheduler_state_dict,
     epoch, chunk_number, scenes_processed, RNG states, validation_type,
     and history records.
4. Continuous Weight Evolution:
   - All chunks update the EXACT same model weights continuously.
5. Strict Data Quarantine:
   - Excludes corrupted TIFFs (00815, 00816, 00817).
   - Test split (180 scenes) remains 100% quarantined and untouched.
"""

import os
import sys
import gc
import time
import copy
import shutil
import random
import argparse
import ctypes
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models_exp2 import build_exp2_segformer_b0
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetricsMeter

# Known corrupt/unreadable training TIFFs that must be excluded
EXCLUDED_TRAIN_TIFFS = {"00815.tif", "00816.tif", "00817.tif"}

# Fixed number of validation scenes used for diagnostic quick validation
QUICK_VAL_SCENE_COUNT = 20


def get_system_ram_gb() -> Dict[str, float]:
    """Queries Windows physical memory statistics via ctypes."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_gb": stat.ullTotalPhys / (1024 ** 3),
        "avail_gb": stat.ullAvailPhys / (1024 ** 3),
        "used_gb": (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3),
        "load_pct": float(stat.dwMemoryLoad),
    }


def load_patch_windowed(
    img_path: str,
    mask_path: str,
    x: int,
    y: int,
    patch_size: int = 256,
    vh_min: float = -50.0,
    vh_max: float = -10.0,
    vv_min: float = -35.0,
    vv_max: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streams a single 256x256 window directly from disk via rasterio.
    Avoids reading the 2048x2048 scene into memory.
    """
    window = Window(x, y, patch_size, patch_size)

    # 1. Read SAR bands (Band 1 = VH, Band 2 = VV)
    with rasterio.open(img_path) as src:
        raw_sar = src.read([1, 2], window=window).astype(np.float32)  # [2, H, W]

    # 2. Read companion mask
    with rasterio.open(mask_path) as m_src:
        raw_mask = m_src.read(1, window=window).astype(np.float32)  # [H, W]

    # 3. Clean NaNs/Infs
    raw_sar = np.nan_to_num(raw_sar, nan=vh_min, posinf=vh_max, neginf=vh_min)
    raw_mask = np.nan_to_num(raw_mask, nan=0.0, posinf=1.0, neginf=0.0)

    # 4. Normalize SAR dB to [0, 1]
    vh = np.clip(raw_sar[0], vh_min, vh_max)
    vh_norm = (vh - vh_min) / (vh_max - vh_min)

    vv = np.clip(raw_sar[1], vv_min, vv_max)
    vv_norm = (vv - vv_min) / (vv_max - vv_min)

    # Convert to PyTorch tensors with batch dimension [1, C, H, W]
    sar_tensor = torch.from_numpy(np.stack([vh_norm, vv_norm], axis=0)).unsqueeze(0)  # [1, 2, 256, 256]
    mask_bin = (raw_mask > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]

    return sar_tensor, mask_tensor


def get_deterministic_val_offsets(num_patches: int = 2, scene_dim: int = 2048, patch_size: int = 256) -> List[Tuple[int, int]]:
    """Generates fixed, reproducible coordinate offsets for validation scenes."""
    max_offset = scene_dim - patch_size
    if num_patches == 1:
        return [(max_offset // 2, max_offset // 2)]
    elif num_patches == 2:
        return [(max_offset // 3, max_offset // 3), (2 * max_offset // 3, 2 * max_offset // 3)]
    elif num_patches == 4:
        return [
            (max_offset // 4, max_offset // 4),
            (3 * max_offset // 4, max_offset // 4),
            (max_offset // 4, 3 * max_offset // 4),
            (3 * max_offset // 4, 3 * max_offset // 4),
        ]
    else:
        offsets = []
        step = max_offset // max(1, int(np.sqrt(num_patches)))
        for y in range(0, max_offset + 1, max(1, step)):
            for x in range(0, max_offset + 1, max(1, step)):
                offsets.append((x, y))
                if len(offsets) >= num_patches:
                    return offsets
        return offsets[:num_patches]


def determine_validation_type(chunk_number: int, total_chunks: int = 42, force_val: Optional[str] = None) -> str:
    """
    Returns the validation type according to schedule:
    - Chunk 1: full (completed)
    - Chunks 2-9: quick
    - Chunk 10: full
    - Chunks 11-19: quick
    - Chunk 20: full
    - Chunks 21-29: quick
    - Chunk 30: full
    - Chunks 31-41: quick
    - Chunk 42: full
    """
    if force_val and force_val.lower() != "auto":
        return force_val.lower()
    
    # Milestone chunks for full validation
    if chunk_number == 1 or chunk_number % 10 == 0 or chunk_number == total_chunks:
        return "full"
    return "quick"


def train_single_chunk(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    chunk_df: pd.DataFrame,
    images_dir: str,
    masks_dir: str,
    patches_per_scene: int = 4,
    device: torch.device = torch.device("cpu"),
    chunk_number: int = 1,
) -> Dict[str, float]:
    """
    Trains on a single discrete chunk of scenes sequentially.
    Streams one scene at a time and applies memory garbage collection.
    """
    model.train()
    meter = SegmentationMetricsMeter(threshold=0.5)
    
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    batch_count = 0
    
    max_coord = 2048 - 256
    t_start = time.perf_counter()

    num_scenes = len(chunk_df)
    print(f"\n--- Training Chunk {chunk_number} ({num_scenes} scenes, {num_scenes * patches_per_scene} patches) ---", flush=True)

    for s_idx, (_, row) in enumerate(chunk_df.iterrows()):
        fname = row["filename"]
        img_p = os.path.join(images_dir, fname)
        mask_p = os.path.join(masks_dir, fname)

        if not (os.path.exists(img_p) and os.path.exists(mask_p)):
            print(f"  [WARNING] Missing file pair for {fname}, skipping.", flush=True)
            continue

        for p_idx in range(patches_per_scene):
            # Random offset for training diversity
            rx = random.randint(0, max_coord)
            ry = random.randint(0, max_coord)

            # 1. Windowed read directly from disk
            img_b, mask_b = load_patch_windowed(img_p, mask_p, rx, ry, patch_size=256)
            img_b = img_b.to(device)
            mask_b = mask_b.to(device)

            # 2. Forward pass
            logits = model(img_b)

            # 3. Loss calculation
            loss, breakdown = criterion(logits, mask_b)

            # 4. Backward pass
            optimizer.zero_grad()
            loss.backward()

            # 5. Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            meter.update(logits, mask_b)
            total_loss += loss.item()
            total_bce += breakdown["bce_loss"]
            total_dice += breakdown["dice_loss"]
            batch_count += 1

            # Release tensor references
            del img_b, mask_b, logits, loss

        # Run garbage collection after each scene to release GDAL/rasterio buffers
        gc.collect()

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == num_scenes:
            avg_loss_so_far = total_loss / max(1, batch_count)
            print(f"  Processed {s_idx + 1}/{num_scenes} scenes ({batch_count} patches) | Running Loss: {avg_loss_so_far:.4f}", flush=True)

    elapsed = time.perf_counter() - t_start
    train_metrics = meter.compute()

    return {
        "train_loss": total_loss / max(1, batch_count),
        "train_bce": total_bce / max(1, batch_count),
        "train_dice": total_dice / max(1, batch_count),
        "train_iou": train_metrics["iou"],
        "train_recall": train_metrics["recall"],
        "train_precision": train_metrics["precision"],
        "batches_processed": batch_count,
        "chunk_time_sec": elapsed,
    }


def evaluate_validation(
    model: nn.Module,
    criterion: nn.Module,
    val_df: pd.DataFrame,
    images_dir: str,
    masks_dir: str,
    val_type: str = "quick",  # "quick" or "full"
    val_patches_per_scene: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluates model on validation set:
    - Quick: Exactly the first 20 fixed scenes of val_df (40 patches, ~1 min)
    - Full: All 180 scenes of val_df (360 patches, ~8.7 min)
    Never touches the quarantined test set.
    """
    model.eval()
    meter = SegmentationMetricsMeter(threshold=0.5)
    
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    batch_count = 0

    if val_type == "quick":
        val_scenes = val_df.iloc[:QUICK_VAL_SCENE_COUNT].reset_index(drop=True)
        desc_label = f"QUICK Validation (Fixed {len(val_scenes)} scenes, {len(val_scenes) * val_patches_per_scene} patches, Diagnostic)"
    else:
        val_scenes = val_df.reset_index(drop=True)
        desc_label = f"FULL Validation ({len(val_scenes)} scenes, {len(val_scenes) * val_patches_per_scene} patches, Official Benchmark)"

    val_offsets = get_deterministic_val_offsets(val_patches_per_scene)
    
    t_start = time.perf_counter()
    print(f"\n--- Evaluating {desc_label} ---", flush=True)

    with torch.no_grad():
        for s_idx, (_, row) in enumerate(val_scenes.iterrows()):
            fname = row["filename"]
            img_p = os.path.join(images_dir, fname)
            mask_p = os.path.join(masks_dir, fname)

            if not (os.path.exists(img_p) and os.path.exists(mask_p)):
                continue

            for x_off, y_off in val_offsets:
                img_b, mask_b = load_patch_windowed(img_p, mask_p, x_off, y_off, patch_size=256)
                img_b = img_b.to(device)
                mask_b = mask_b.to(device)

                logits = model(img_b)
                loss, breakdown = criterion(logits, mask_b)

                meter.update(logits, mask_b)
                total_loss += loss.item()
                total_bce += breakdown["bce_loss"]
                total_dice += breakdown["dice_loss"]
                batch_count += 1

                del img_b, mask_b, logits, loss

            gc.collect()

    elapsed = time.perf_counter() - t_start
    val_metrics = meter.compute()
    val_metrics["val_loss"] = total_loss / max(1, batch_count)
    val_metrics["val_bce"] = total_bce / max(1, batch_count)
    val_metrics["val_dice_loss"] = total_dice / max(1, batch_count)
    val_metrics["val_eval_time_sec"] = elapsed
    val_metrics["val_type"] = val_type
    val_metrics["val_scenes_count"] = len(val_scenes)

    print(f"  [{val_type.upper()} VAL] Loss: {val_metrics['val_loss']:.4f} | IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f} | Recall: {val_metrics['recall']:.4f} ({elapsed:.1f}s)", flush=True)
    return val_metrics


def save_chunk_checkpoint(
    checkpoint_dir: str,
    chunk_number: int,
    epoch: int,
    scenes_start: int,
    scenes_end: int,
    scenes_processed: int,
    total_usable_scenes: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    train_results: Dict[str, float],
    val_results: Dict[str, float],
    val_type: str,
    best_val_iou: float,
    training_history: List[Dict[str, Any]],
    history_path: str,
    total_elapsed_seconds: float,
) -> float:
    """Saves atomic chunk checkpoint and updates latest.pth, best.pth, and history CSV."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)

    ram_stats = get_system_ram_gb()
    
    # SCIENTIFIC ACCURACY: Only FULL validation can claim a new official best model!
    is_new_best = False
    if val_type == "full":
        if val_results.get("iou", 0.0) > best_val_iou:
            best_val_iou = val_results.get("iou", 0.0)
            is_new_best = True

    checkpoint_data = {
        "chunk_number": chunk_number,
        "epoch": epoch,
        "scenes_start": scenes_start,
        "scenes_end": scenes_end,
        "scenes_processed": scenes_processed,
        "total_usable_scenes": total_usable_scenes,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "best_val_iou": best_val_iou,
        "train_metrics": train_results,
        "validation_metrics": val_results,
        "validation_type": val_type,
        "random_state": {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
        },
        "training_history": training_history,
        "ram_stats": ram_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": "SegFormer-B0 (mit_b0)",
            "in_channels": 2,
            "classes": 1,
            "patch_size": 256,
        },
    }

    # 1. Save numbered chunk checkpoint
    chunk_fname = f"chunk_{chunk_number:03d}.pth"
    chunk_path = os.path.join(checkpoint_dir, chunk_fname)
    torch.save(checkpoint_data, chunk_path)
    print(f"  [CHECKPOINT] Saved numbered checkpoint: {chunk_path}", flush=True)

    # 2. Save latest.pth
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(checkpoint_data, latest_path)
    print(f"  [CHECKPOINT] Updated latest checkpoint: {latest_path}", flush=True)

    # 3. Save best.pth if and only if FULL validation achieved new highest IoU
    if is_new_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(checkpoint_data, best_path)
        print(f"  [CHECKPOINT] New best FULL validation IoU ({best_val_iou:.4f})! Saved: {best_path}", flush=True)
    elif val_type == "quick":
        print(f"  [DIAGNOSTIC] Quick validation IoU: {val_results.get('iou', 0.0):.4f} (Diagnostic only; best.pth remains at {best_val_iou:.4f})", flush=True)

    # 4. Record to training_history.csv with all requested columns
    record = {
        "chunk": chunk_number,
        "scenes_start": scenes_start,
        "scenes_end": scenes_end,
        "train_loss": train_results.get("train_loss", 0.0),
        "train_iou": train_results.get("train_iou", 0.0),
        "val_type": val_type,
        "val_loss": val_results.get("val_loss", 0.0),
        "val_iou": val_results.get("iou", 0.0),
        "val_dice": val_results.get("dice", 0.0),
        "val_recall": val_results.get("recall", 0.0),
        "elapsed_seconds": round(total_elapsed_seconds, 2),
        # Supplementary metadata columns
        "epoch": epoch,
        "scenes_processed": scenes_processed,
        "ram_avail_gb": round(ram_stats["avail_gb"], 2),
        "timestamp": checkpoint_data["timestamp"],
    }
    
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        # Normalize column name if previous version had chunk_number instead of chunk
        if "chunk_number" in hist_df.columns and "chunk" not in hist_df.columns:
            hist_df.rename(columns={"chunk_number": "chunk"}, inplace=True)
        # Remove any existing row for this chunk to allow clean re-runs
        hist_df = hist_df[hist_df["chunk"] != chunk_number]
        hist_df = pd.concat([hist_df, pd.DataFrame([record])], ignore_index=True)
    else:
        hist_df = pd.DataFrame([record])

    # Reorder columns to ensure requested columns appear first
    priority_cols = [
        "chunk", "scenes_start", "scenes_end", "train_loss", "train_iou",
        "val_type", "val_loss", "val_iou", "val_dice", "val_recall", "elapsed_seconds"
    ]
    all_cols = priority_cols + [c for c in hist_df.columns if c not in priority_cols]
    hist_df = hist_df[all_cols]
    
    hist_df.to_csv(history_path, index=False)
    print(f"  [HISTORY] Updated training log: {history_path}", flush=True)

    return best_val_iou


def load_resumable_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
) -> Tuple[int, int, int, float, List[Dict[str, Any]]]:
    """
    Restores model, optimizer, scheduler, and RNG state from checkpoint.
    Returns: (last_chunk_number, current_epoch, scenes_processed, best_val_iou, training_history)
    """
    print(f"  [RESUME] Loading checkpoint from: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG state for reproducible continuation
    if "random_state" in checkpoint:
        rng = checkpoint["random_state"]
        random.setstate(rng["python_random"])
        np.random.set_state(rng["numpy_random"])
        torch.set_rng_state(rng["torch_random"])

    chunk_number = checkpoint.get("chunk_number", 0)
    epoch = checkpoint.get("epoch", 1)
    scenes_processed = checkpoint.get("scenes_processed", 0)
    best_val_iou = checkpoint.get("best_val_iou", 0.0)
    training_history = checkpoint.get("training_history", [])

    print(f"  [RESUME] Successfully restored Chunk {chunk_number} (Epoch {epoch}, {scenes_processed} scenes processed, Best Val IoU: {best_val_iou:.4f})", flush=True)
    return chunk_number, epoch, scenes_processed, best_val_iou, training_history


def run_chunk_training(
    chunk_size: int = 20,
    target_chunk: Optional[int] = None,
    single_chunk_only: bool = False,
    max_chunks_to_run: Optional[int] = None,
    resume: bool = True,
    resume_checkpoint_path: Optional[str] = None,
    checkpoint_dir: str = "outputs/checkpoints",
    history_path: str = "outputs/training_history.csv",
    images_dir: str = "extracted_dataset/images",
    masks_dir: str = "extracted_dataset/masks",
    patches_per_scene: int = 4,
    val_patches_per_scene: int = 2,
    force_val: Optional[str] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dry_run: bool = False,
):
    """
    Main chunk execution controller.
    """
    print("=" * 65, flush=True)
    print("OCEANTRACE EXPERIMENT 2: RESUMABLE CHUNK-TRAINING ENGINE", flush=True)
    print("=" * 65, flush=True)
    
    # 1. System checks
    ram_init = get_system_ram_gb()
    print(f"System RAM: {ram_init['avail_gb']:.2f} GB available / {ram_init['total_gb']:.2f} GB total ({ram_init['load_pct']}% load)", flush=True)

    # 2. Load splits & filter excluded scenes
    train_csv_path = "data/splits/experiment2_train.csv"
    val_csv_path = "data/splits/experiment2_val.csv"
    
    df_train_raw = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Strictly exclude the 3 known corrupted training TIFFs
    df_train = df_train_raw[~df_train_raw["filename"].isin(EXCLUDED_TRAIN_TIFFS)].reset_index(drop=True)
    total_usable = len(df_train)
    print(f"Dataset Splits Verified:", flush=True)
    print(f"  - Usable Training Scenes: {total_usable} (excluded {len(EXCLUDED_TRAIN_TIFFS)} known unreadable files: {sorted(list(EXCLUDED_TRAIN_TIFFS))})", flush=True)
    print(f"  - Validation Scenes:      {len(df_val)} (Quick val uses first {QUICK_VAL_SCENE_COUNT} fixed scenes)", flush=True)
    print(f"  - Test Scenes (QUARANTINED): 180 scenes strictly preserved and untouched", flush=True)

    # 3. Calculate chunk structure
    total_chunks = (total_usable + chunk_size - 1) // chunk_size
    print(f"Chunk Configuration:", flush=True)
    print(f"  - Chunk Size:             {chunk_size} scenes/chunk", flush=True)
    print(f"  - Total Chunks:           {total_chunks} chunks per epoch", flush=True)
    print(f"  - Patches per Scene:      {patches_per_scene} (Train), {val_patches_per_scene} (Validation)", flush=True)
    print(f"  - Batch Size:             1 (Memory-Safe CPU Streaming)", flush=True)
    print("=" * 65, flush=True)

    if dry_run:
        print("[DRY RUN] Verifying all chunk boundaries and validation schedule:", flush=True)
        for c in range(1, total_chunks + 1):
            s_start = (c - 1) * chunk_size
            s_end = min(c * chunk_size, total_usable)
            v_type = determine_validation_type(c, total_chunks, force_val)
            sub = df_train.iloc[s_start:s_end]
            print(f"  Chunk {c:02d}: scenes {s_start + 1:03d} to {s_end:03d} ({len(sub)} scenes: {sub.iloc[0]['filename']} .. {sub.iloc[-1]['filename']}) -> Val: {v_type.upper()}", flush=True)
        print("[DRY RUN] Completed successfully. Exiting without training.", flush=True)
        return

    # 4. Instantiate Model, Loss, Optimizer
    device = torch.device("cpu")
    model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # 5. Check for resuming
    latest_pth = os.path.join(checkpoint_dir, "latest.pth")
    resume_target = resume_checkpoint_path or (latest_pth if os.path.exists(latest_pth) and resume else None)

    last_completed_chunk = 0
    current_epoch = 1
    scenes_processed = 0
    best_val_iou = 0.0
    training_history = []

    if resume_target and os.path.exists(resume_target):
        last_completed_chunk, current_epoch, scenes_processed, best_val_iou, training_history = load_resumable_checkpoint(
            resume_target, model, optimizer, scheduler
        )

    # Determine starting chunk
    if target_chunk is not None:
        start_chunk = target_chunk
    else:
        start_chunk = last_completed_chunk + 1

    if start_chunk > total_chunks:
        print(f"All {total_chunks} chunks for Epoch {current_epoch} are completed! Incrementing epoch...", flush=True)
        current_epoch += 1
        start_chunk = 1
        scenes_processed = 0

    print(f"\nExecution Target:", flush=True)
    print(f"  Starting Chunk:   Chunk {start_chunk} (Epoch {current_epoch})", flush=True)
    if single_chunk_only:
        print(f"  Execution Mode:   Single Chunk Only (Chunk {start_chunk})", flush=True)
        end_chunk = start_chunk
    elif max_chunks_to_run is not None:
        end_chunk = min(start_chunk + max_chunks_to_run - 1, total_chunks)
        print(f"  Execution Mode:   Running {max_chunks_to_run} Chunks (Chunks {start_chunk} to {end_chunk})", flush=True)
    else:
        end_chunk = total_chunks
        print(f"  Execution Mode:   Continuous through end of epoch (Chunks {start_chunk} to {end_chunk})", flush=True)

    # 6. Execute chunk loop
    for chunk_idx in range(start_chunk, end_chunk + 1):
        s_start = (chunk_idx - 1) * chunk_size
        s_end = min(chunk_idx * chunk_size, total_usable)
        chunk_slice = df_train.iloc[s_start:s_end].reset_index(drop=True)

        val_type = determine_validation_type(chunk_idx, total_chunks, force_val)

        print(f"\n" + "#" * 65, flush=True)
        print(f"# EXECUTING CHUNK {chunk_idx}/{total_chunks} (Scenes {s_start + 1} to {s_end}) | Val: {val_type.upper()}", flush=True)
        print("#" * 65, flush=True)

        chunk_t0 = time.perf_counter()

        # Train on chunk
        train_res = train_single_chunk(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            chunk_df=chunk_slice,
            images_dir=images_dir,
            masks_dir=masks_dir,
            patches_per_scene=patches_per_scene,
            device=device,
            chunk_number=chunk_idx,
        )
        scenes_processed += len(chunk_slice)

        # Validation evaluation according to schedule
        if val_type != "none":
            val_res = evaluate_validation(
                model=model,
                criterion=criterion,
                val_df=df_val,
                images_dir=images_dir,
                masks_dir=masks_dir,
                val_type=val_type,
                val_patches_per_scene=val_patches_per_scene,
                device=device,
            )
        else:
            val_res = {"iou": 0.0, "dice": 0.0, "recall": 0.0, "val_loss": 0.0, "val_type": "none"}

        # Step learning rate scheduler if chunk completed an epoch
        if chunk_idx == total_chunks:
            scheduler.step()

        total_chunk_time = time.perf_counter() - chunk_t0

        # Save checkpoint
        best_val_iou = save_chunk_checkpoint(
            checkpoint_dir=checkpoint_dir,
            chunk_number=chunk_idx,
            epoch=current_epoch,
            scenes_start=s_start + 1,
            scenes_end=s_end,
            scenes_processed=scenes_processed,
            total_usable_scenes=total_usable,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_results=train_res,
            val_results=val_res,
            val_type=val_type,
            best_val_iou=best_val_iou,
            training_history=training_history,
            history_path=history_path,
            total_elapsed_seconds=total_chunk_time,
        )

        ram_cur = get_system_ram_gb()
        print(f"Completed Chunk {chunk_idx} in {total_chunk_time:.1f}s (Train: {train_res['chunk_time_sec']:.1f}s, Val: {val_res.get('val_eval_time_sec', 0.0):.1f}s) | Current RAM Available: {ram_cur['avail_gb']:.2f} GB ({ram_cur['load_pct']}% load)", flush=True)

        if single_chunk_only:
            break

    print("\n" + "=" * 65, flush=True)
    print("CHUNK EXECUTION RUN COMPLETED", flush=True)
    print(f"Checkpoints directory: {os.path.abspath(checkpoint_dir)}", flush=True)
    print(f"Training history file: {os.path.abspath(history_path)}", flush=True)
    print("=" * 65, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="OceanTrace CPU Resumable Chunk-Training Engine (Optimized)")
    parser.add_argument("--chunk_size", type=int, default=20, help="Number of scenes per chunk (default: 20)")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Specific chunk to run (default: auto-resume next)")
    parser.add_argument("--single_chunk", action="store_true", help="Execute exactly one chunk and exit")
    parser.add_argument("--max_chunks", type=int, default=None, help="Maximum number of chunks to run in this invocation")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from latest checkpoint (start from chunk 1)")
    parser.add_argument("--resume_path", type=str, default=None, help="Explicit checkpoint file path to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--history_path", type=str, default="outputs/training_history.csv", help="CSV path for training logs")
    parser.add_argument("--patches_per_scene", type=int, default=4, help="Number of patches per training scene (default: 4)")
    parser.add_argument("--val_patches", type=int, default=2, help="Number of patches per validation scene (default: 2)")
    parser.add_argument("--force_val", type=str, default="auto", choices=["auto", "full", "quick", "none"],
                        help="Validation mode: auto (follows schedule), full (180 scenes), quick (20 fixed scenes), or none")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--dry_run", action="store_true", help="Test chunk boundaries and verification without training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_chunk_training(
        chunk_size=args.chunk_size,
        target_chunk=args.chunk_idx,
        single_chunk_only=args.single_chunk,
        max_chunks_to_run=args.max_chunks,
        resume=not args.no_resume,
        resume_checkpoint_path=args.resume_path,
        checkpoint_dir=args.checkpoint_dir,
        history_path=args.history_path,
        patches_per_scene=args.patches_per_scene,
        val_patches_per_scene=args.val_patches,
        force_val=args.force_val,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dry_run=args.dry_run,
    )
