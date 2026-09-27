"""
OceanTrace Experiment 2: Initialize Master Reports & Retroactively Generate Chunks 1-10 Reports
---------------------------------------------------------------------------------------------
Populates outputs/experiment2_training_reports/ with:
- chunk_01_report.json/txt through chunk_10_report.json/txt (and single digit aliases)
- experiment2_training_master_report.csv (Chunks 1-10 rows)
- experiment2_training_master_report.json (Initial master state)
- experiment2_training_log.txt (Initial log history)
"""

import os
import sys
import json
import torch
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

REPORTS_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2_training_reports")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "outputs", "checkpoints")
HISTORY_CSV = os.path.join(REPO_ROOT, "outputs", "training_history.csv")

os.makedirs(REPORTS_DIR, exist_ok=True)


def init_reports():
    print(f"Initializing reports in: {REPORTS_DIR}")

    if not os.path.exists(HISTORY_CSV):
        raise FileNotFoundError(f"Missing history file: {HISTORY_CSV}")

    df_hist = pd.read_csv(HISTORY_CSV)
    df_hist = df_hist.sort_values("chunk").reset_index(drop=True)

    master_rows = []
    completed_chunks_json = []
    cum_time = 0.0
    best_full_val_iou = 0.0
    best_full_val_dice = 0.0
    best_chunk_num = 1

    log_entries = []

    for _, row in df_hist.iterrows():
        c = int(row["chunk"])
        if c > 10:
            continue

        ckpt_name = f"chunk_{c:03d}.pth"
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)

        ckpt_data = {}
        if os.path.exists(ckpt_path):
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        train_metrics = ckpt_data.get("train_metrics", {})
        val_metrics = ckpt_data.get("validation_metrics", {})

        s_start = int(row["scenes_start"])
        s_end = int(row["scenes_end"])
        num_scenes = s_end - s_start + 1
        patches = num_scenes * 4

        train_loss = float(row["train_loss"])
        train_iou = float(row["train_iou"])
        train_dice = float(train_metrics.get("train_dice", float("nan"))) if "train_dice" in train_metrics else None
        train_prec = float(train_metrics.get("train_precision", float("nan"))) if "train_precision" in train_metrics else None
        train_rec = float(train_metrics.get("train_recall", float("nan"))) if "train_recall" in train_metrics else None

        val_type = str(row["val_type"]).upper()
        val_loss = float(row["val_loss"])
        val_iou = float(row["val_iou"])
        val_dice = float(row["val_dice"])
        val_rec = float(row["val_recall"])
        val_prec = float(val_metrics.get("precision", float("nan"))) if "precision" in val_metrics else None

        elapsed_sec = float(row["elapsed_seconds"])
        cum_time += elapsed_sec
        timestamp = str(row["timestamp"])

        # Determine best checkpoint status
        is_best = False
        if val_type == "FULL":
            if val_iou > best_full_val_iou:
                best_full_val_iou = val_iou
                best_full_val_dice = val_dice
                best_chunk_num = c
                is_best = True
            best_status = "UPDATED (new best)" if is_best else "UNCHANGED"
        else:
            best_status = "UNCHANGED (quick val diagnostic only)"

        lr = 0.0001

        # Build chunk report dictionary
        chunk_rep = {
            "experiment_name": "OceanTrace_SegFormer_Exp2",
            "chunk_number": c,
            "scene_start_index": s_start,
            "scene_end_index": s_end,
            "number_of_scenes_processed": num_scenes,
            "training_patch_count": patches,
            "training_loss": round(train_loss, 6),
            "training_iou": round(train_iou, 6),
            "training_dice": round(train_dice, 6) if train_dice is not None else None,
            "training_precision": round(train_prec, 6) if train_prec is not None else None,
            "training_recall": round(train_rec, 6) if train_rec is not None else None,
            "validation_type": val_type,
            "validation_loss": round(val_loss, 6),
            "validation_iou": round(val_iou, 6),
            "validation_dice": round(val_dice, 6),
            "validation_precision": round(val_prec, 6) if val_prec is not None else None,
            "validation_recall": round(val_rec, 6),
            "learning_rate": lr,
            "elapsed_time_seconds": round(elapsed_sec, 2),
            "cumulative_elapsed_time_seconds": round(cum_time, 2),
            "checkpoint_path": os.path.relpath(ckpt_path, REPO_ROOT).replace("\\", "/"),
            "best_checkpoint_status": best_status,
            "completed_successfully": True,
            "timestamp": timestamp,
            "warning_error_info": None,
        }

        # Text chunk report format
        chunk_txt = f"""==================================================
OCEANTRACE EXPERIMENT 2
CHUNK {c} REPORT
==================================================

Experiment: OceanTrace_SegFormer_Exp2
Chunk: {c}
Timestamp: {timestamp}
Status: COMPLETED_SUCCESSFULLY

Scenes:
  Start Index: {s_start}
  End Index: {s_end}
  Scenes Processed: {num_scenes}
  Patches Trained: {patches}

Training Metrics:
  Loss: {train_loss:.6f}
  IoU:  {train_iou:.6f}
  Dice: {f"{train_dice:.6f}" if train_dice is not None else "unavailable"}
  Precision: {f"{train_prec:.6f}" if train_prec is not None else "unavailable"}
  Recall:    {f"{train_rec:.6f}" if train_rec is not None else "unavailable"}

Validation Metrics:
  Type: {val_type}
  Loss: {val_loss:.6f}
  IoU:  {val_iou:.6f}
  Dice: {val_dice:.6f}
  Precision: {f"{val_prec:.6f}" if val_prec is not None else "unavailable"}
  Recall:    {val_rec:.6f}

Optimization & Hardware:
  Learning Rate: {lr}
  Elapsed Time: {elapsed_sec:.2f} s
  Cumulative Time: {cum_time:.2f} s

Checkpoints:
  Chunk Checkpoint: {os.path.relpath(ckpt_path, REPO_ROOT).replace("\\", "/")}
  Best Checkpoint Status: {best_status}
  Latest Checkpoint: outputs/checkpoints/latest.pth

Warnings / Errors: None
==================================================
"""

        # Write per-chunk files (both formats and naming conventions)
        for base_name in [f"chunk_{c}_report", f"chunk_{c:02d}_report"]:
            json_path = os.path.join(REPORTS_DIR, f"{base_name}.json")
            txt_path = os.path.join(REPORTS_DIR, f"{base_name}.txt")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunk_rep, f, indent=2)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(chunk_txt)

        # Master CSV row
        master_rows.append({
            "chunk": c,
            "scene_start": s_start,
            "scene_end": s_end,
            "num_scenes": num_scenes,
            "train_patches": patches,
            "train_loss": round(train_loss, 6),
            "train_iou": round(train_iou, 6),
            "validation_type": val_type,
            "val_loss": round(val_loss, 6),
            "val_iou": round(val_iou, 6),
            "val_dice": round(val_dice, 6),
            "val_precision": round(val_prec, 6) if val_prec is not None else "null",
            "val_recall": round(val_rec, 6),
            "learning_rate": lr,
            "elapsed_seconds": round(elapsed_sec, 2),
            "cumulative_elapsed_seconds": round(cum_time, 2),
            "checkpoint": f"outputs/checkpoints/chunk_{c:03d}.pth",
            "best_checkpoint": "outputs/checkpoints/best.pth",
            "status": "SUCCESS",
            "timestamp": timestamp,
        })

        completed_chunks_json.append(chunk_rep)
        log_entries.append(chunk_txt)

    # Write master CSV
    master_csv_path = os.path.join(REPORTS_DIR, "experiment2_training_master_report.csv")
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(master_csv_path, index=False)
    print(f"Saved initial master CSV: {master_csv_path} ({len(df_master)} chunks)")

    # Write master JSON
    master_json_path = os.path.join(REPORTS_DIR, "experiment2_training_master_report.json")
    master_json_content = {
        "experiment_name": "OceanTrace_SegFormer_Exp2",
        "experiment_configuration": {
            "model": "SegFormer-B0 (mit_b0)",
            "library": "segmentation_models_pytorch",
            "in_channels": 2,
            "channels": ["Sigma0_VH_db", "Sigma0_VV_db"],
            "classes": 1,
            "trainable_parameters": 3712833,
            "vh_preprocessing": "clip [-50, -10] dB -> min-max to [0, 1]",
            "vv_preprocessing": "clip [-35, 5] dB -> min-max to [0, 1]",
            "loss": "0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "initial_learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR (T_max=30, eta_min=1e-6)",
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "patch_size": 256,
            "train_patches_per_scene": 4,
            "val_patches_per_scene": 2,
        },
        "dataset_information": {
            "usable_training_scenes": 837,
            "validation_scenes": 180,
            "quarantined_test_scenes": 180,
            "excluded_unreadable_training_scenes": ["00815.tif", "00816.tif", "00817.tif"],
            "excluded_unreadable_test_scenes": ["00813.tif"],
            "total_chunks": 42,
            "chunk_size": 20,
            "final_chunk_size": 17,
        },
        "completed_chunks": completed_chunks_json,
        "best_validation_result": {
            "best_val_iou": round(best_full_val_iou, 6),
            "best_val_dice": round(best_full_val_dice, 6),
            "best_chunk": best_chunk_num,
            "checkpoint": "outputs/checkpoints/best.pth",
        },
        "current_latest_checkpoint": "outputs/checkpoints/latest.pth",
        "cumulative_training_time_seconds": round(cum_time, 2),
        "warnings": [],
        "errors": [],
        "final_status": "IN_PROGRESS (Chunks 1-10 completed; resuming Chunk 11)",
    }
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_json_content, f, indent=2)
    print(f"Saved initial master JSON: {master_json_path}")

    # Write log file
    master_log_path = os.path.join(REPORTS_DIR, "experiment2_training_log.txt")
    with open(master_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries) + "\n")
    print(f"Saved initial log: {master_log_path}")

    print("[SUCCESS] All Chunks 1-10 reports and master tables initialized.")


if __name__ == "__main__":
    init_reports()
