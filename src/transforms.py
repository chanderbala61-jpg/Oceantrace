"""
Transforms and Preprocessing for Sentinel-1 SAR Oil Spill Data
--------------------------------------------------------------
Applies SAR decibel clipping, min-max normalization, and
controlled geometric augmentations (Rot90, HorizontalFlip, VerticalFlip).
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def preprocess_sar_db(img: np.ndarray, db_min: float = -35.0, db_max: float = 5.0) -> np.ndarray:
    """
    Clip SAR decibel (dB) backscatter to [db_min, db_max] and scale to [0.0, 1.0].
    
    Args:
        img: Single-band float32 SAR array (H, W) or (H, W, 1) in dB.
        db_min: Minimum dB threshold (calm water/thick slick floor).
        db_max: Maximum dB threshold (strong reflectors ceiling).
        
    Returns:
        Normalized float32 array in [0.0, 1.0].
    """
    img_clipped = np.clip(img, db_min, db_max)
    img_norm = (img_clipped - db_min) / (db_max - db_min)
    return img_norm.astype(np.float32)


def get_train_transforms(patch_size: int = 256):
    """
    Controlled augmentation pipeline for Experiment 1:
      - RandomHorizontalFlip (p=0.5)
      - RandomVerticalFlip (p=0.5)
      - RandomRotate90 (p=0.5)
    Note: Gaussian speckle noise is strictly disabled for Experiment 1 baseline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])


def get_val_transforms():
    """
    Deterministic transformation for validation and testing (ToTensor only).
    """
    return A.Compose([
        ToTensorV2()
    ])
