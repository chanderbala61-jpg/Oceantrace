"""
Dataset & Patch Extraction Module for Sentinel-1 SAR Oil Spill Segmentation
--------------------------------------------------------------------------
Handles scene-level loading, 256x256 patch extraction, SAR dB normalization,
and PyTorch DataLoader construction.
"""

import os
import glob
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio

from src.transforms import preprocess_sar_db, get_train_transforms, get_val_transforms


class SAROilSpillDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR Oil Spill 256x256 Patches.
    """
    def __init__(
        self,
        patch_records: List[Dict],
        transforms=None,
        db_min: float = -35.0,
        db_max: float = 5.0,
    ):
        """
        Args:
            patch_records: List of dicts containing {'image_patch': np.ndarray, 'mask_patch': np.ndarray, 'scene_name': str}
                           or paths/coordinates for on-demand extraction.
            transforms: Albumentations Compose pipeline.
            db_min: Minimum dB value for clipping.
            db_max: Maximum dB value for clipping.
        """
        self.records = patch_records
        self.transforms = transforms
        self.db_min = db_min
        self.db_max = db_max

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        img_patch = rec["image_patch"]  # float32 dB (H, W)
        mask_patch = rec["mask_patch"]  # float32 binary (H, W)

        # Preprocess SAR dB
        img_norm = preprocess_sar_db(img_patch, self.db_min, self.db_max)

        # Albumentations expects (H, W, C) for image and (H, W) for mask
        img_hwc = np.expand_dims(img_norm, axis=-1)  # (256, 256, 1)

        if self.transforms is not None:
            augmented = self.transforms(image=img_hwc, mask=mask_patch)
            img_tensor = augmented["image"]  # (1, 256, 256) float32
            mask_tensor = augmented["mask"]  # (256, 256)
        else:
            img_tensor = torch.from_numpy(img_hwc).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_patch).float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, 256, 256)

        return {
            "image": img_tensor.float(),
            "mask": mask_tensor.float(),
            "scene_name": rec.get("scene_name", "unknown")
        }


def extract_patches_from_scene(
    img_path: str,
    mask_path: str,
    patch_size: int = 256,
    stride: int = 128,
    min_oil_pixels: int = 10,
    bg_sampling_ratio: float = 0.3,
    is_train: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Extracts 256x256 patches from full GeoTIFF scene image and mask.
    """
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing file: {img_path} or {mask_path}")

    img = tifffile.imread(img_path).astype(np.float32)
    mask = tifffile.imread(mask_path).astype(np.float32)

    # Ensure 2D
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    h, w = img.shape
    scene_name = os.path.basename(img_path)

    pos_patches = []
    neg_patches = []

    rng = np.random.RandomState(seed)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = img[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            # Replace NaNs or Infs if any
            img_patch = np.nan_to_num(img_patch, nan=-35.0, posinf=5.0, neginf=-35.0)
            mask_patch = np.nan_to_num(mask_patch, nan=0.0)

            oil_count = np.sum(mask_patch > 0.5)
            record = {
                "image_patch": img_patch,
                "mask_patch": mask_patch,
                "scene_name": scene_name,
                "coords": (y, x),
                "has_oil": bool(oil_count >= min_oil_pixels)
            }

            if oil_count >= min_oil_pixels:
                pos_patches.append(record)
            else:
                neg_patches.append(record)

    if is_train and bg_sampling_ratio < 1.0 and len(neg_patches) > 0:
        # Balanced sampling: keep all positive patches and a controlled fraction of negative patches
        n_keep_neg = max(int(len(pos_patches) * bg_sampling_ratio), min(len(neg_patches), 50))
        shuffled_neg_idx = rng.permutation(len(neg_patches))[:n_keep_neg]
        selected_neg = [neg_patches[i] for i in shuffled_neg_idx]
        all_patches = pos_patches + selected_neg
    else:
        all_patches = pos_patches + neg_patches

    return all_patches


def build_split_dataset(
    split_csv_path: str,
    raw_data_dir: str,
    patch_size: int = 256,
    stride: int = 128,
    is_train: bool = True,
    bg_sampling_ratio: float = 0.3,
    min_oil_pixels: int = 10,
    db_min: float = -35.0,
    db_max: float = 5.0,
    seed: int = 42,
    smoke_test: bool = False,
    smoke_test_samples: int = 16
) -> SAROilSpillDataset:
    """
    Builds a SAROilSpillDataset for a given split CSV manifest.
    """
    df_scenes = pd.read_csv(split_csv_path)
    all_patch_records = []

    print(f"Loading {len(df_scenes)} scenes from {split_csv_path}...")
    for _, row in df_scenes.iterrows():
        img_rel = row["relative_img_path"]
        mask_rel = row["relative_mask_path"]
        
        img_path = os.path.join(raw_data_dir, img_rel)
        mask_path = os.path.join(raw_data_dir, mask_rel)

        patches = extract_patches_from_scene(
            img_path=img_path,
            mask_path=mask_path,
            patch_size=patch_size,
            stride=stride,
            min_oil_pixels=min_oil_pixels,
            bg_sampling_ratio=bg_sampling_ratio,
            is_train=is_train,
            seed=seed
        )
        all_patch_records.extend(patches)
        print(f"  - Scene {row['scene_name']}: extracted {len(patches)} patches (oil/bg balanced)")
        
        if smoke_test and len(all_patch_records) >= smoke_test_samples:
            break

    if smoke_test:
        all_patch_records = all_patch_records[:smoke_test_samples]
        print(f"[SMOKE TEST] Trimmed dataset to {len(all_patch_records)} samples.")

    transforms = get_train_transforms(patch_size) if is_train else get_val_transforms()
    return SAROilSpillDataset(
        patch_records=all_patch_records,
        transforms=transforms,
        db_min=db_min,
        db_max=db_max
    )


def get_dataloaders(
    config: dict,
    raw_data_dir: Optional[str] = None,
    smoke_test: bool = False,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Creates Train, Validation, and optionally Test PyTorch DataLoaders.
    """
    data_dir = raw_data_dir or config["data"]["raw_dir"]
    splits_dir = config["data"]["splits_dir"]
    patch_size = config["data"]["patch_size"]
    train_stride = config["data"]["train_stride"]
    val_stride = config["data"]["val_stride"]
    db_min = config["data"]["db_clip_min"]
    db_max = config["data"]["db_clip_max"]
    seed = config["project"]["seed"]
    bs = batch_size or config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 2)
    pin_memory = config["data"].get("pin_memory", True)

    train_split_csv = os.path.join(splits_dir, "train_scenes.csv")
    val_split_csv = os.path.join(splits_dir, "val_scenes.csv")

    train_ds = build_split_dataset(
        split_csv_path=train_split_csv,
        raw_data_dir=data_dir,
        patch_size=patch_size,
        stride=train_stride,
        is_train=True,
        bg_sampling_ratio=config["data"]["bg_sampling_ratio"],
        min_oil_pixels=config["data"]["min_oil_pixels"],
        db_min=db_min,
        db_max=db_max,
        seed=seed,
        smoke_test=smoke_test,
        smoke_test_samples=16 if smoke_test else None
    )

    val_ds = build_split_dataset(
        split_csv_path=val_split_csv,
        raw_data_dir=data_dir,
        patch_size=patch_size,
        stride=val_stride,
        is_train=False,
        bg_sampling_ratio=1.0,
        min_oil_pixels=config["data"]["min_oil_pixels"],
        db_min=db_min,
        db_max=db_max,
        seed=seed,
        smoke_test=smoke_test,
        smoke_test_samples=8 if smoke_test else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader
