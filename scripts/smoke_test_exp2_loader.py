import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from src.datasets.experiment2_sar_dataset import (
    Experiment2SARDataset,
    create_exp2_dataloader,
    get_exp2_train_transforms,
    get_exp2_eval_transforms,
)

def run_smoke_test():
    print("=== STARTING EXPERIMENT 2 DATASET SMOKE TEST ===", flush=True)

    train_csv = "data/splits/experiment2_train.csv"
    val_csv = "data/splits/experiment2_val.csv"
    test_csv = "data/splits/experiment2_test.csv"
    images_dir = "extracted_dataset/images"
    masks_dir = "extracted_dataset/masks"

    # 1. Dataset Initialization
    print("\n1. Initializing Datasets...", flush=True)
    train_ds = Experiment2SARDataset(
        split_csv=train_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
        patch_size=256,
        patches_per_scene=2,
        is_train=True,
        transforms=get_exp2_train_transforms(),
    )
    val_ds = Experiment2SARDataset(
        split_csv=val_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
        patch_size=256,
        patches_per_scene=2,
        is_train=False,
        transforms=get_exp2_eval_transforms(),
    )
    test_ds = Experiment2SARDataset(
        split_csv=test_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
        patch_size=256,
        patches_per_scene=2,
        is_train=False,
        transforms=get_exp2_eval_transforms(),
    )

    print(f"Train dataset length: {len(train_ds)} (scenes: {len(train_ds.df)})", flush=True)
    print(f"Val dataset length:   {len(val_ds)} (scenes: {len(val_ds.df)})", flush=True)
    print(f"Test dataset length:  {len(test_ds)} (scenes: {len(test_ds.df)})", flush=True)

    # 2. Sample Loading Tests
    print("\n2. Loading Single Samples...", flush=True)
    train_sample = train_ds[0]
    val_sample = val_ds[0]
    test_sample = test_ds[0]

    img_t = train_sample["image"]
    mask_t = train_sample["mask"]
    print(f"Train sample img shape:  {img_t.shape}, dtype: {img_t.dtype}", flush=True)
    print(f"Train sample mask shape: {mask_t.shape}, dtype: {mask_t.dtype}", flush=True)
    print(f"Val sample img shape:    {val_sample['image'].shape}", flush=True)
    print(f"Test sample img shape:   {test_sample['image'].shape}", flush=True)

    # Shape assertions
    assert img_t.shape == torch.Size([2, 256, 256]), f"Expected [2, 256, 256], got {img_t.shape}"
    assert mask_t.shape == torch.Size([1, 256, 256]), f"Expected [1, 256, 256], got {mask_t.shape}"
    assert val_sample["image"].shape == torch.Size([2, 256, 256]), f"Expected [2, 256, 256], got {val_sample['image'].shape}"
    assert test_sample["image"].shape == torch.Size([2, 256, 256]), f"Expected [2, 256, 256], got {test_sample['image'].shape}"

    # 3. Finite Values & Stats
    vh_channel = img_t[0].numpy()
    vv_channel = img_t[1].numpy()

    assert np.isfinite(vh_channel).all(), "VH channel contains NaN or Inf!"
    assert np.isfinite(vv_channel).all(), "VV channel contains NaN or Inf!"
    assert np.isfinite(mask_t.numpy()).all(), "Mask contains NaN or Inf!"

    # 4. Check that VH and VV channels contain non-identical data
    channel_diff = np.abs(vh_channel - vv_channel).sum()
    print(f"\n3. Dual-Polarization SAR Channel Check:", flush=True)
    print(f"VH min: {vh_channel.min():.4f}, max: {vh_channel.max():.4f}, mean: {vh_channel.mean():.4f}", flush=True)
    print(f"VV min: {vv_channel.min():.4f}, max: {vv_channel.max():.4f}, mean: {vv_channel.mean():.4f}", flush=True)
    print(f"Absolute sum difference between VH and VV: {channel_diff:.4f}", flush=True)
    assert channel_diff > 1.0, "VH and VV channels appear identical!"

    # 5. Mask Binary Values
    mask_unique = torch.unique(mask_t).tolist()
    print(f"\n4. Mask unique values: {mask_unique}", flush=True)
    for u in mask_unique:
        assert u in [0.0, 1.0], f"Unexpected mask value: {u}"

    # 6. DataLoader Batch Verification
    print("\n5. Testing PyTorch DataLoader batch production...", flush=True)
    loader = create_exp2_dataloader(
        split_csv=train_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=4,
        is_train=True,
        patch_size=256,
        patches_per_scene=2,
        num_workers=0,
    )

    batch = next(iter(loader))
    batch_img = batch["image"]
    batch_mask = batch["mask"]
    print(f"Batch image shape: {batch_img.shape}", flush=True)
    print(f"Batch mask shape:  {batch_mask.shape}", flush=True)
    print(f"Batch filenames:   {batch['filename']}", flush=True)

    assert batch_img.shape == torch.Size([4, 2, 256, 256]), f"Expected [4, 2, 256, 256], got {batch_img.shape}"
    assert batch_mask.shape == torch.Size([4, 1, 256, 256]), f"Expected [4, 1, 256, 256], got {batch_mask.shape}"

    print("\n=== SMOKE TEST RESULT: ALL 14 VERIFICATIONS PASSED ===", flush=True)

if __name__ == "__main__":
    run_smoke_test()
