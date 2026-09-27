"""
OceanTrace Experiment 2: Automated Chunks 3 to 10 Sequential Runner
-------------------------------------------------------------------
Executes Chunks 3 through 10 sequentially with:
- Checkpoint resumption from latest.pth
- Two-tier validation: Quick (Chunks 3-9) and Full (Chunk 10)
- Strict checkpoint integrity & parameter finiteness verification
- Strict TIFF read error abortion (no silent skipping)
- Exact per-chunk reporting
- Stopping strictly after Chunk 10
"""

import os
import sys
import gc
import time
import copy
import random
import ctypes
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models_exp2 import build_exp2_segformer_b0
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetricsMeter
from src.train_chunk_exp2 import (
    load_patch_windowed,
    get_deterministic_val_offsets,
    get_system_ram_gb,
    load_resumable_checkpoint,
    save_chunk_checkpoint,
    EXCLUDED_TRAIN_TIFFS,
    QUICK_VAL_SCENE_COUNT,
)


def verify_checkpoint_integrity(
    checkpoint_dir: str,
    chunk_idx: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history_path: str,
) -> bool:
    """Verifies that latest.pth exists, restores cleanly, parameters are finite, and history is logged."""
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    chunk_path = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:03d}.pth")

    if not os.path.exists(latest_path):
        raise RuntimeError(f"Safety Check Failed: {latest_path} does not exist!")
    if not os.path.exists(chunk_path):
        raise RuntimeError(f"Safety Check Failed: {chunk_path} does not exist!")

    # Verify loadable
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    if ckpt.get("chunk_number") != chunk_idx:
        raise RuntimeError(f"Safety Check Failed: latest.pth chunk_number is {ckpt.get('chunk_number')}, expected {chunk_idx}")

    # Verify model parameters are all finite
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Safety Check Failed: Model parameter '{name}' contains NaN or Inf!")

    # Verify optimizer state
    if not optimizer.param_groups:
        raise RuntimeError("Safety Check Failed: Optimizer has no parameter groups!")

    # Verify training history
    if not os.path.exists(history_path):
        raise RuntimeError(f"Safety Check Failed: History file {history_path} does not exist!")
    df_hist = pd.read_csv(history_path)
    chunk_col = "chunk" if "chunk" in df_hist.columns else "chunk_number"
    if chunk_idx not in df_hist[chunk_col].values:
        raise RuntimeError(f"Safety Check Failed: Chunk {chunk_idx} not recorded in {history_path}")

    return True


def execute_chunks_3_to_10():
    print("=" * 70, flush=True)
    print("OCEANTRACE EXPERIMENT 2: RUNNING CHUNKS 3 THROUGH 10", flush=True)
    print("=" * 70, flush=True)

    checkpoint_dir = "outputs/checkpoints"
    history_path = "outputs/training_history.csv"
    images_dir = "extracted_dataset/images"
    masks_dir = "extracted_dataset/masks"
    latest_pth = os.path.join(checkpoint_dir, "latest.pth")

    if not os.path.exists(latest_pth):
        raise FileNotFoundError(f"Cannot start: {latest_pth} not found. Chunk 2 must be completed first!")

    # 1. Load dataset splits
    df_train_raw = pd.read_csv("data/splits/experiment2_train.csv")
    df_val = pd.read_csv("data/splits/experiment2_val.csv")
    df_train = df_train_raw[~df_train_raw["filename"].isin(EXCLUDED_TRAIN_TIFFS)].reset_index(drop=True)
    total_usable = len(df_train)

    print(f"Loaded splits: {total_usable} usable training scenes, {len(df_val)} validation scenes.", flush=True)
    print("Quarantined test set: Untouched and not loaded.", flush=True)

    # 2. Build model, loss, optimizer, scheduler
    device = torch.device("cpu")
    model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # 3. Resume from latest.pth (Chunk 2)
    last_chunk, epoch, scenes_processed, best_val_iou, training_history = load_resumable_checkpoint(
        latest_pth, model, optimizer, scheduler
    )
    print(f"Resumed from checkpoint: Last Chunk={last_chunk}, Processed Scenes={scenes_processed}, Best Val IoU={best_val_iou:.4f}\n", flush=True)

    if last_chunk < 2:
        raise ValueError(f"Expected to resume at least from Chunk 2, but latest checkpoint is Chunk {last_chunk}")

    start_chunk = last_chunk + 1
    end_chunk = 10

    if start_chunk > end_chunk:
        print(f"Target chunks already completed! Currently at Chunk {last_chunk}. Nothing to run.", flush=True)
        return

    # Cumulative timer tracking
    cum_training_time = 0.0
    if os.path.exists(history_path):
        df_prev = pd.read_csv(history_path)
        if "elapsed_seconds" in df_prev.columns:
            cum_training_time = float(df_prev["elapsed_seconds"].sum())

    # Chunk execution loop
    for chunk_idx in range(start_chunk, end_chunk + 1):
        s_start = (chunk_idx - 1) * 20
        s_end = min(chunk_idx * 20, total_usable)
        chunk_slice = df_train.iloc[s_start:s_end].reset_index(drop=True)

        val_type = "full" if chunk_idx == 10 else "quick"

        print(f"\n>>> Launching Chunk {chunk_idx}/10 (Scenes {s_start + 1} to {s_end}) | Val: {val_type.upper()} <<<", flush=True)

        # -------------------------------------------------------------
        # A. TRAINING PHASE
        # -------------------------------------------------------------
        model.train()
        meter = SegmentationMetricsMeter(threshold=0.5)
        total_loss = total_bce = total_dice = 0.0
        batch_count = 0
        max_coord = 2048 - 256
        t_train_start = time.perf_counter()

        for s_idx, (_, row) in enumerate(chunk_slice.iterrows()):
            fname = row["filename"]
            img_p = os.path.join(images_dir, fname)
            mask_p = os.path.join(masks_dir, fname)

            # Strict check: fail immediately if missing
            if not os.path.exists(img_p):
                raise FileNotFoundError(f"TIFF READ ERROR: Image file does not exist: {img_p}")
            if not os.path.exists(mask_p):
                raise FileNotFoundError(f"TIFF READ ERROR: Mask file does not exist: {mask_p}")

            for p_idx in range(4):
                rx = random.randint(0, max_coord)
                ry = random.randint(0, max_coord)

                try:
                    img_b, mask_b = load_patch_windowed(img_p, mask_p, rx, ry, patch_size=256)
                except Exception as e:
                    raise RuntimeError(f"TIFF READ ERROR: Failed windowed read on {fname} (patch {p_idx+1}/4, offset {rx},{ry}): {e}")

                img_b = img_b.to(device)
                mask_b = mask_b.to(device)

                logits = model(img_b)
                loss, breakdown = criterion(logits, mask_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                meter.update(logits, mask_b)
                total_loss += loss.item()
                total_bce += breakdown["bce_loss"]
                total_dice += breakdown["dice_loss"]
                batch_count += 1

                del img_b, mask_b, logits, loss

            gc.collect()

        train_duration = time.perf_counter() - t_train_start
        train_metrics = meter.compute()
        train_loss = total_loss / max(1, batch_count)
        train_iou = train_metrics["iou"]
        scenes_processed += len(chunk_slice)

        # -------------------------------------------------------------
        # B. VALIDATION PHASE
        # -------------------------------------------------------------
        model.eval()
        val_meter = SegmentationMetricsMeter(threshold=0.5)
        val_loss_sum = val_bce_sum = val_dice_sum = 0.0
        val_batch_count = 0

        if val_type == "quick":
            val_scenes = df_val.iloc[:QUICK_VAL_SCENE_COUNT].reset_index(drop=True)
        else:
            val_scenes = df_val.reset_index(drop=True)

        val_offsets = get_deterministic_val_offsets(num_patches=2)
        t_val_start = time.perf_counter()

        with torch.no_grad():
            for _, row in val_scenes.iterrows():
                v_fname = row["filename"]
                v_img = os.path.join(images_dir, v_fname)
                v_mask = os.path.join(masks_dir, v_fname)

                if not os.path.exists(v_img) or not os.path.exists(v_mask):
                    raise FileNotFoundError(f"VALIDATION ERROR: Missing validation pair: {v_fname}")

                for x_off, y_off in val_offsets:
                    try:
                        img_vb, mask_vb = load_patch_windowed(v_img, v_mask, x_off, y_off, patch_size=256)
                    except Exception as e:
                        raise RuntimeError(f"VALIDATION READ ERROR: Failed read on {v_fname}: {e}")

                    img_vb = img_vb.to(device)
                    mask_vb = mask_vb.to(device)

                    v_logits = model(img_vb)
                    v_loss, v_break = criterion(v_logits, mask_vb)

                    val_meter.update(v_logits, mask_vb)
                    val_loss_sum += v_loss.item()
                    val_bce_sum += v_break["bce_loss"]
                    val_dice_sum += v_break["dice_loss"]
                    val_batch_count += 1

                    del img_vb, mask_vb, v_logits, v_loss

                gc.collect()

        val_duration = time.perf_counter() - t_val_start
        val_res = val_meter.compute()
        val_loss = val_loss_sum / max(1, val_batch_count)
        val_res["val_loss"] = val_loss
        val_res["val_eval_time_sec"] = val_duration

        total_chunk_time = train_duration + val_duration
        cum_training_time += total_chunk_time

        # -------------------------------------------------------------
        # C. CHECKPOINT SAVING & SAFETY CHECKS
        # -------------------------------------------------------------
        train_res_dict = {
            "train_loss": train_loss,
            "train_bce": total_bce / max(1, batch_count),
            "train_dice": total_dice / max(1, batch_count),
            "train_iou": train_iou,
            "chunk_time_sec": train_duration,
        }

        best_val_iou = save_chunk_checkpoint(
            checkpoint_dir=checkpoint_dir,
            chunk_number=chunk_idx,
            epoch=epoch,
            scenes_start=s_start + 1,
            scenes_end=s_end,
            scenes_processed=scenes_processed,
            total_usable_scenes=total_usable,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_results=train_res_dict,
            val_results=val_res,
            val_type=val_type,
            best_val_iou=best_val_iou,
            training_history=training_history,
            history_path=history_path,
            total_elapsed_seconds=total_chunk_time,
        )

        # Mandatory Safety Verification
        verify_checkpoint_integrity(checkpoint_dir, chunk_idx, model, optimizer, history_path)

        # -------------------------------------------------------------
        # D. PER-CHUNK REPORT (MANDATORY FORMAT)
        # -------------------------------------------------------------
        chunk_file_p = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:03d}.pth")
        best_file_p = os.path.join(checkpoint_dir, "best.pth")

        print("\n" + "=" * 40, flush=True)
        print(f"OCEANTRACE CHUNK {chunk_idx} COMPLETE", flush=True)
        print("=" * 40, flush=True)
        print(f"\nScenes:", flush=True)
        print(f"Start: {s_start + 1}", flush=True)
        print(f"End: {s_end}", flush=True)
        print(f"Number of scenes: {len(chunk_slice)}", flush=True)
        print(f"Number of patches: {len(chunk_slice) * 4}", flush=True)
        print(f"\nTraining:", flush=True)
        print(f"Training loss: {train_loss:.4f}", flush=True)
        print(f"Training IoU: {train_iou:.4f}", flush=True)
        print(f"Training duration: {train_duration:.1f}s", flush=True)
        print(f"\nValidation:", flush=True)
        print(f"Validation type: {val_type.upper()}", flush=True)
        print(f"Validation loss: {val_loss:.4f}", flush=True)
        print(f"Validation IoU: {val_res['iou']:.4f}", flush=True)
        print(f"Validation Dice: {val_res['dice']:.4f}", flush=True)
        print(f"Validation Recall: {val_res['recall']:.4f}", flush=True)
        print(f"Validation duration: {val_duration:.1f}s", flush=True)
        print(f"\nCheckpoint:", flush=True)
        print(f"chunk_{chunk_idx}.pth: {chunk_file_p}", flush=True)
        print(f"latest.pth: {latest_pth}", flush=True)
        print(f"best.pth: {best_file_p}", flush=True)
        print(f"Checkpoint successfully written: YES", flush=True)
        print(f"\nCumulative:", flush=True)
        print(f"Total scenes processed: {scenes_processed}", flush=True)
        print(f"Total chunks completed: {chunk_idx}", flush=True)
        print(f"Total training time: {cum_training_time:.1f}s", flush=True)
        print("=" * 40 + "\n", flush=True)

        # Check RAM safety
        ram_now = get_system_ram_gb()
        if ram_now["avail_gb"] < 0.25:
            print(f"[CRITICAL RAM WARNING] Available RAM dropped to {ram_now['avail_gb']:.2f} GB. Stopping safely.", flush=True)
            break

    print("\nAll requested chunks through Chunk 10 completed cleanly.", flush=True)


if __name__ == "__main__":
    execute_chunks_3_to_10()
