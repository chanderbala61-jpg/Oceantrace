"""
Experiment 2 Two-Channel Sentinel-1 SAR Oil Spill Dataset & Patch Extraction Module
-----------------------------------------------------------------------------------
Preserves the dual-polarization SAR channels (Band 0 = VH, Band 1 = VV) as a 2-channel
tensor [2, H, W] without RGB duplication or channel dropping.
Performs dynamic, memory-efficient on-demand patch extraction.
"""

import os
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd
import rasterio
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_dual_channel_sar(
    img: np.ndarray,
    vh_min: float = -50.0,
    vh_max: float = -10.0,
    vv_min: float = -35.0,
    vv_max: float = 5.0,
) -> np.ndarray:
    """
    Normalizes 2-channel SAR backscatter [H, W, 2] in decibels (dB) to [0.0, 1.0].
    Channel 0 = Sigma0_VH_db
    Channel 1 = Sigma0_VV_db

    Args:
        img: np.ndarray of shape (H, W, 2), float32.
        vh_min: Lower dB threshold for VH cross-polarization.
        vh_max: Upper dB threshold for VH cross-polarization.
        vv_min: Lower dB threshold for VV co-polarization (matching Exp 1 baseline).
        vv_max: Upper dB threshold for VV co-polarization (matching Exp 1 baseline).

    Returns:
        np.ndarray of shape (H, W, 2), float32 in [0.0, 1.0].
    """
    # Defensive NoData / NaN / Inf sanitization
    img = np.nan_to_num(img, nan=vh_min, posinf=vh_max, neginf=vh_min)

    vh = np.clip(img[..., 0], vh_min, vh_max)
    vh_norm = (vh - vh_min) / (vh_max - vh_min)

    vv = np.clip(img[..., 1], vv_min, vv_max)
    vv_norm = (vv - vv_min) / (vv_max - vv_min)

    return np.stack([vh_norm, vv_norm], axis=-1).astype(np.float32)


class Experiment2SARDataset(Dataset):
    """
    Two-channel dynamic patch extraction dataset for Experiment 2.
    Loads Sentinel-1 GeoTIFFs (VH and VV bands) and verified companion masks {0, 1}.
    Supports both unified flat folders and partitioned Part_01..Part_10 directory structures.
    """

    def __init__(
        self,
        split_csv: str,
        images_dir: str = "extracted_dataset/images",
        masks_dir: str = "extracted_dataset/masks",
        patch_size: int = 256,
        patches_per_scene: int = 4,
        is_train: bool = True,
        transforms=None,
        vh_min: float = -50.0,
        vh_max: float = -10.0,
        vv_min: float = -35.0,
        vv_max: float = 5.0,
        seed: int = 42,
    ):
        """
        Args:
            split_csv: Path to experiment2_train.csv / val.csv / test.csv
            images_dir: Directory containing 2048x2048 2-band image TIFFs (or root containing Part_01..Part_10)
            masks_dir: Directory containing 2048x2048 binary mask TIFFs (or root containing Part_01..Part_10)
            patch_size: Square patch dimension (default 256)
            patches_per_scene: Number of patches extracted per scene per epoch
            is_train: If True, uses random patch sampling; if False, deterministic grid
            transforms: Albumentations composition for spatial augmentations
            vh_min, vh_max, vv_min, vv_max: Radiometric clipping boundaries
            seed: RNG seed for reproducible validation/test patch sampling
        """
        self.df = pd.read_csv(split_csv)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patch_size = patch_size
        self.patches_per_scene = patches_per_scene
        self.is_train = is_train
        self.transforms = transforms
        self.vh_min = vh_min
        self.vh_max = vh_max
        self.vv_min = vv_min
        self.vv_max = vv_max
        self.rng = np.random.RandomState(seed)

        # Build dynamic file index to support both flat directories and Part_01..Part_10 partitions
        self._file_map = self._build_file_map()

        # Precompute deterministic coordinates for eval modes
        if not self.is_train:
            self.eval_coords = self._generate_eval_coords()

    def _build_file_map(self) -> Dict[str, Tuple[str, str]]:
        """Builds path resolution map supporting flat and Part_01..Part_10 partitioned structures."""
        file_map = {}
        if len(self.df) == 0:
            return file_map

        # Check if first file exists directly in images_dir / masks_dir
        first_file = str(self.df.iloc[0]["filename"])
        direct_img = os.path.join(self.images_dir, first_file)
        direct_mask = os.path.join(self.masks_dir, first_file)

        if os.path.exists(direct_img) and os.path.exists(direct_mask):
            for f in self.df["filename"]:
                file_map[f] = (
                    os.path.join(self.images_dir, f),
                    os.path.join(self.masks_dir, f),
                )
            return file_map

        # Search across candidate directories for Part_01..Part_10
        search_roots = [
            self.images_dir,
            self.masks_dir,
            os.path.dirname(self.images_dir.rstrip("/\\")),
            os.path.dirname(os.path.dirname(self.images_dir.rstrip("/\\"))),
            "Oceantrace_Colab_Exp2_UploadParts",
            ".",
        ]
        found_images = {}
        found_masks = {}

        seen_roots = set()
        for root in search_roots:
            if not root or not os.path.isdir(root) or root in seen_roots:
                continue
            seen_roots.add(root)
            for current_dir, _, files in os.walk(root):
                norm_dir = current_dir.replace("\\", "/").lower()
                is_img_dir = "images" in norm_dir
                is_mask_dir = "masks" in norm_dir
                if not (is_img_dir or is_mask_dir):
                    continue
                for fname in files:
                    if fname.endswith(".tif"):
                        full_p = os.path.join(current_dir, fname)
                        if is_img_dir and fname not in found_images:
                            found_images[fname] = full_p
                        elif is_mask_dir and fname not in found_masks:
                            found_masks[fname] = full_p

        for f in self.df["filename"]:
            img_p = found_images.get(f, os.path.join(self.images_dir, f))
            mask_p = found_masks.get(f, os.path.join(self.masks_dir, f))
            file_map[f] = (img_p, mask_p)

        return file_map

    def _generate_eval_coords(self) -> List[Tuple[int, int]]:
        """Generates deterministic center/grid offsets within a 2048x2048 scene."""
        max_idx = 2048 - self.patch_size
        coords = []
        step = max(1, max_idx // max(1, int(np.sqrt(self.patches_per_scene))))
        for y in range(0, max_idx + 1, step):
            for x in range(0, max_idx + 1, step):
                coords.append((y, x))
                if len(coords) >= self.patches_per_scene:
                    return coords
        while len(coords) < self.patches_per_scene:
            coords.append((0, 0))
        return coords[: self.patches_per_scene]

    def __len__(self) -> int:
        return len(self.df) * self.patches_per_scene

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        scene_idx = idx // self.patches_per_scene
        patch_sub_idx = idx % self.patches_per_scene

        row = self.df.iloc[scene_idx]
        filename = row["filename"]

        img_path, mask_path = self._file_map[filename]

        # 1. Read 2-band SAR GeoTIFF via rasterio
        with rasterio.open(img_path) as src:
            # src.read() returns (bands, H, W) -> transpose to (H, W, 2)
            # Band 1 = VH, Band 2 = VV
            img_raw = src.read([1, 2]).transpose(1, 2, 0).astype(np.float32)

        # 2. Read single-band companion mask via tifffile
        mask_raw = tifffile.imread(mask_path).astype(np.float32)

        # Defensive binarization: ensure {0, 255} or multi-level is mapped to strictly {0.0, 1.0}
        if np.max(mask_raw) > 1.0:
            mask_raw = (mask_raw > 0.5).astype(np.float32)

        # 3. Patch Extraction
        H, W = mask_raw.shape
        max_y = max(0, H - self.patch_size)
        max_x = max(0, W - self.patch_size)

        if self.is_train:
            y0 = np.random.randint(0, max_y + 1)
            x0 = np.random.randint(0, max_x + 1)
        else:
            y0, x0 = self.eval_coords[patch_sub_idx]

        img_patch = img_raw[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size, :]
        mask_patch = mask_raw[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        # 4. Radiometric normalization: [H, W, 2] in [0.0, 1.0]
        img_norm = preprocess_dual_channel_sar(
            img_patch,
            vh_min=self.vh_min,
            vh_max=self.vh_max,
            vv_min=self.vv_min,
            vv_max=self.vv_max,
        )

        # 5. Spatial Augmentation (identical transformations on img and mask)
        if self.transforms is not None:
            augmented = self.transforms(image=img_norm, mask=mask_patch)
            img_tensor = augmented["image"]  # (2, patch_size, patch_size)
            mask_tensor = augmented["mask"]  # (patch_size, patch_size)
        else:
            # (H, W, 2) -> (2, H, W)
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_patch).float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, patch_size, patch_size)

        return {
            "image": img_tensor.float(),
            "mask": mask_tensor.float(),
            "filename": filename,
            "parent_acquisition_id": row.get("parent_acquisition_id", "unknown"),
        }


def get_exp2_train_transforms():
    """Controlled, safe geometric augmentations preserving multi-channel SAR alignment."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])


def get_exp2_eval_transforms():
    """Deterministic validation/test transform."""
    return A.Compose([
        ToTensorV2(),
    ])


def create_exp2_dataloader(
    split_csv: str,
    images_dir: str = "extracted_dataset/images",
    masks_dir: str = "extracted_dataset/masks",
    batch_size: int = 4,
    is_train: bool = True,
    patch_size: int = 256,
    patches_per_scene: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Convenience constructor for Experiment 2 DataLoader."""
    transforms = get_exp2_train_transforms() if is_train else get_exp2_eval_transforms()
    dataset = Experiment2SARDataset(
        split_csv=split_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
        patch_size=patch_size,
        patches_per_scene=patches_per_scene,
        is_train=is_train,
        transforms=transforms,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False,
    )
