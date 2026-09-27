"""
Evaluation Pipeline for OceanTrace SegFormer Experiment 1
---------------------------------------------------------
Evaluates trained SegFormer model on the 7 official held-out test scenes.
Computes global and per-scene metrics (IoU, Dice, Precision, Recall, TP, FP, FN, TN).
"""

import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import tifffile
from typing import Dict, List, Any

from src.models import build_segformer_b0
from src.transforms import preprocess_sar_db
from src.metrics import SegmentationMetricsMeter


@torch.no_grad()
def evaluate_full_scene(
    model: torch.nn.Module,
    img_path: str,
    mask_path: str,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 256,
    db_min: float = -35.0,
    db_max: float = 5.0,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Runs sliding window inference on a full GeoTIFF scene.
    """
    model.eval()
    img = tifffile.imread(img_path).astype(np.float32)
    mask = tifffile.imread(mask_path).astype(np.float32)

    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    h, w = img.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Sliding window inference
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            patch_norm = preprocess_sar_db(patch, db_min, db_max)
            tensor = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)
            
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            prob_map[y : y + patch_size, x : x + patch_size] += probs
            count_map[y : y + patch_size, x : x + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    final_prob = prob_map / count_map
    pred_binary = (final_prob >= threshold).astype(np.float32)
    gt_binary = (mask >= 0.5).astype(np.float32)

    # Compute scene confusion matrix
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))

    eps = 1e-7
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "scene_name": os.path.basename(img_path),
        "shape": (h, w),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pred_binary": pred_binary,
        "final_prob": final_prob,
        "gt_binary": gt_binary,
        "sar_img": img
    }


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    raw_data_dir: Optional[str] = None,
    allow_cpu: bool = False
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow_cpu for local dev testing.")

    data_dir = raw_data_dir or cfg["data"]["raw_dir"]
    splits_dir = cfg["data"]["splits_dir"]
    test_split_csv = os.path.join(splits_dir, "test_scenes.csv")
    df_test = pd.read_csv(test_split_csv)

    print("==================================================")
    print("RUNNING FINAL TEST EVALUATION")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Device: {device}")
    print(f"  - Test scenes ({len(df_test)}): {df_test['scene_name'].tolist()}")
    print("==================================================")

    model = build_segformer_b0(
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        encoder_weights=None,
        verbose=False
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[PASS] Checkpoint loaded successfully.")

    scene_results = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for _, row in df_test.iterrows():
        img_path = os.path.join(data_dir, row["relative_img_path"])
        mask_path = os.path.join(data_dir, row["relative_mask_path"])

        res = evaluate_full_scene(
            model=model,
            img_path=img_path,
            mask_path=mask_path,
            device=device,
            patch_size=cfg["data"]["patch_size"],
            stride=cfg["data"]["val_stride"],
            db_min=cfg["data"]["db_clip_min"],
            db_max=cfg["data"]["db_clip_max"],
            threshold=cfg["training"]["threshold"]
        )

        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]
        total_tn += res["tn"]

        scene_results.append({
            "scene_name": res["scene_name"],
            "iou": res["iou"],
            "dice": res["dice"],
            "precision": res["precision"],
            "recall": res["recall"],
            "tp": res["tp"],
            "fp": res["fp"],
            "fn": res["fn"],
            "tn": res["tn"]
        })

        print(f"Scene: {res['scene_name']} | IoU: {res['iou']:.4f}, Dice: {res['dice']:.4f}, Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}")

    # Global aggregate metrics
    eps = 1e-7
    global_iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    global_dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    global_precision = (total_tp + eps) / (total_tp + total_fp + eps)
    global_recall = (total_tp + eps) / (total_tp + total_fn + eps)
    global_acc = (total_tp + total_tn + eps) / (total_tp + total_tn + total_fp + total_fn + eps)

    summary = {
        "global_iou": float(global_iou),
        "global_dice": float(global_dice),
        "global_precision": float(global_precision),
        "global_recall": float(global_recall),
        "global_accuracy": float(global_acc),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_tn": int(total_tn),
        "threshold": cfg["training"]["threshold"],
        "checkpoint": checkpoint_path,
        "per_scene_metrics": scene_results
    }

    log_dir = cfg["outputs"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "test_evaluation_summary.json")
    scene_csv_path = os.path.join(log_dir, "test_metrics_per_scene.csv")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(scene_results).to_csv(scene_csv_path, index=False)

    print("==================================================")
    print("GLOBAL TEST METRICS SUMMARY")
    print("==================================================")
    print(f"  - Overall IoU (Jaccard):   {global_iou:.4f}")
    print(f"  - Overall Dice (F1-Score): {global_dice:.4f}")
    print(f"  - Overall Precision:       {global_precision:.4f}")
    print(f"  - Overall Recall:          {global_recall:.4f}")
    print(f"  - Overall Accuracy:        {global_acc:.4f}")
    print(f"  - Confusion: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
    print(f"\nSaved test reports to:\n  - {summary_path}\n  - {scene_csv_path}")
    print("==================================================")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegFormer-B0 on Test Scenes")
    parser.add_argument("--config", type=str, default="configs/exp1_segformer_b0.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        raw_data_dir=args.data_dir,
        allow_cpu=args.allow_cpu
    )
