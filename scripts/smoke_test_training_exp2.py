import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.experiment2_sar_dataset import (
    Experiment2SARDataset,
    get_exp2_train_transforms,
    get_exp2_eval_transforms,
)
from src.models_exp2 import build_exp2_segformer_b0
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetricsMeter

def run_training_smoke_test():
    print("=== STARTING EXPERIMENT 2 TRAINING SMOKE TEST ===", flush=True)

    # 1. Device & Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 2. Output directory
    checkpoint_dir = "outputs/experiment2_segformer_b0/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Model Architecture
    model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    first_conv_weight = model.encoder.patch_embed1.proj.weight
    assert first_conv_weight.shape[1] == 2, f"Expected 2 input channels in patch projection, got {first_conv_weight.shape[1]}"

    # 4. Data Loaders (using real files from experiment2_train.csv and experiment2_val.csv)
    train_ds = Experiment2SARDataset(
        split_csv="data/splits/experiment2_train.csv",
        images_dir="extracted_dataset/images",
        masks_dir="extracted_dataset/masks",
        patch_size=256,
        patches_per_scene=1,
        is_train=True,
        transforms=get_exp2_train_transforms(),
    )
    val_ds = Experiment2SARDataset(
        split_csv="data/splits/experiment2_val.csv",
        images_dir="extracted_dataset/images",
        masks_dir="extracted_dataset/masks",
        patch_size=256,
        patches_per_scene=1,
        is_train=False,
        transforms=get_exp2_eval_transforms(),
    )

    # Subset to 4 samples for quick 2-batch smoke test
    train_loader = DataLoader(Subset(train_ds, [0, 1, 2, 3]), batch_size=2, shuffle=True)
    val_loader = DataLoader(Subset(val_ds, [0, 1]), batch_size=2, shuffle=False)

    # 5. Loss & Optimizer
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 6. Training Step (Forward, Loss, Backward, Step)
    model.train()
    train_losses = []
    print("\nExecuting Training Batches...", flush=True)

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        print(f"  Batch {step+1}: input image shape={images.shape}, mask shape={masks.shape}", flush=True)
        assert images.shape == torch.Size([2, 2, 256, 256]), f"Unexpected image shape {images.shape}"
        assert masks.shape == torch.Size([2, 1, 256, 256]), f"Unexpected mask shape {masks.shape}"

        optimizer.zero_grad()
        logits = model(images)
        assert logits.shape == torch.Size([2, 1, 256, 256]), f"Unexpected logits shape {logits.shape}"

        loss, breakdown = criterion(logits, masks)
        assert torch.isfinite(loss), f"Non-finite loss encountered: {loss.item()}"
        print(f"  Batch {step+1} Loss: {loss.item():.4f} (BCE: {breakdown['bce_loss']:.4f}, Dice: {breakdown['dice_loss']:.4f})", flush=True)

        loss.backward()

        # Check gradients are finite
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

        optimizer.step()
        train_losses.append(loss.item())

    print("All training steps succeeded with finite gradients and optimizer step!", flush=True)

    # 7. Validation Step
    print("\nExecuting Validation Evaluation...", flush=True)
    model.eval()
    val_meter = SegmentationMetricsMeter(threshold=0.5)
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss, _ = criterion(logits, masks)
            val_losses.append(loss.item())
            val_meter.update(logits, masks)

    val_metrics = val_meter.compute()
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Validation Loss: {avg_val_loss:.4f}", flush=True)
    print(f"Validation Metrics: IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}", flush=True)

    # 8. Checkpoint Save & Reload Verification
    checkpoint_path = os.path.join(checkpoint_dir, "smoke_test_checkpoint.pt")
    checkpoint_data = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou": val_metrics["iou"],
        "train_loss": train_losses[-1],
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"\nSaved checkpoint to: {checkpoint_path} ({os.path.getsize(checkpoint_path)} bytes)", flush=True)

    # Reload into a fresh model
    reloaded_model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
    reloaded_model.to(device)
    loaded_ckpt = torch.load(checkpoint_path, map_location=device)
    reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    reloaded_model.eval()

    # Verify identical output on sample
    with torch.no_grad():
        orig_out = model(images)
        reloaded_out = reloaded_model(images)
        diff = torch.abs(orig_out - reloaded_out).sum().item()
        print(f"Reload verification: difference between original and reloaded logits = {diff:.6f}", flush=True)
        assert diff < 1e-5, "Reloaded model produced differing outputs!"

    print("\n=== TRAINING SMOKE TEST RESULT: ALL 11 VERIFICATIONS PASSED ===", flush=True)

if __name__ == "__main__":
    run_training_smoke_test()
