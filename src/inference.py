"""
OceanTrace SegFormer-B0 Inference & SAR Preprocessing Engine
============================================================
Enforces strict model freezing, checkpoint integrity validation,
dual-channel SAR radiometric preprocessing, and deterministic inference.

Guarantees:
- Model weights remain frozen (no fine-tuning, no optimizer steps).
- Checkpoint SHA-256 fingerprint verified against locked benchmark:
  cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6
- Decision threshold locked at 0.5 (identical to final test evaluation).
- CPU and CUDA hardware support with automatic memory management.
"""

import os
import hashlib
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models_exp2 import build_exp2_segformer_b0

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCKED_CHECKPOINT_SHA256 = "cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6"
DEFAULT_CHECKPOINT_PATH = os.path.join(REPO_ROOT, "outputs", "checkpoints", "best.pth")

# Radiometric normalization bounds (SAR dB)
VH_MIN_DB = -50.0
VH_MAX_DB = -10.0
VV_MIN_DB = -35.0
VV_MAX_DB = 5.0
LOCKED_THRESHOLD = 0.5


def verify_checkpoint_hash(filepath: str) -> Tuple[bool, str]:
    """
    Computes SHA-256 hash of checkpoint file and compares to locked hash.
    """
    if not os.path.isfile(filepath):
        return False, "FILE_NOT_FOUND"
    
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return digest == LOCKED_CHECKPOINT_SHA256, digest


def load_frozen_segformer(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    verify_hash: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads the validated SegFormer-B0 model from checkpoint and sets to eval mode.
    
    Returns:
        (model, metadata_dict)
    """
    if checkpoint_path is None:
        ckpt_path = DEFAULT_CHECKPOINT_PATH
    elif os.path.isabs(checkpoint_path):
        ckpt_path = checkpoint_path
    elif os.path.exists(checkpoint_path):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(REPO_ROOT, checkpoint_path)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    is_verified, actual_hash = verify_checkpoint_hash(ckpt_path)
    if verify_hash and not is_verified:
        raise ValueError(
            f"Checkpoint SHA-256 verification failed!\n"
            f"Expected: {LOCKED_CHECKPOINT_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    # Determine execution device
    if device is None or device == "auto":
        exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        exec_device = torch.device(device)

    # Instantiate model
    model = build_exp2_segformer_b0(
        in_channels=2,
        classes=1,
        encoder_weights=None,
        verbose=False
    )

    # Load state dict
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.to(exec_device)
    model.eval()

    metadata = {
        "checkpoint_path": ckpt_path,
        "sha256": actual_hash,
        "hash_verified": is_verified,
        "chunk_number": ckpt_data.get("chunk_number", 42),
        "validation_metrics": ckpt_data.get("validation_metrics", {}),
        "device": str(exec_device),
        "trainable_parameters": 0,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, metadata


def preprocess_sar_imagery(
    sar_image: np.ndarray,
    vh_min: float = VH_MIN_DB,
    vh_max: float = VH_MAX_DB,
    vv_min: float = VV_MIN_DB,
    vv_max: float = VV_MAX_DB,
) -> np.ndarray:
    """
    Normalizes 2-channel Sentinel-1 SAR imagery (dB) to [0, 1].
    
    Args:
        sar_image: ndarray of shape [H, W, 2] with channel 0 = VH, channel 1 = VV.
    
    Returns:
        ndarray of shape [H, W, 2] float32 normalized in [0, 1].
    """
    cleaned = np.nan_to_num(
        sar_image,
        nan=vh_min,
        posinf=vh_max,
        neginf=vh_min
    ).astype(np.float32)

    # Channel 0: VH
    vh = np.clip(cleaned[..., 0], vh_min, vh_max)
    vh_norm = (vh - vh_min) / (vh_max - vh_min + 1e-7)

    # Channel 1: VV
    vv = np.clip(cleaned[..., 1], vv_min, vv_max)
    vv_norm = (vv - vv_min) / (vv_max - vv_min + 1e-7)

    return np.stack([vh_norm, vv_norm], axis=-1)


def predict_scene_tiled(
    model: nn.Module,
    sar_image: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    threshold: float = LOCKED_THRESHOLD,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes memory-efficient patch-windowed inference over a full SAR scene.
    
    Args:
        model: SegFormer-B0 model in eval mode.
        sar_image: [H, W, 2] raw SAR dB image.
        patch_size: Square tile dimension (default 256).
        stride: Stride between tiles (default 256).
        threshold: Decision threshold (default locked 0.5).
        batch_size: Number of tiles per forward pass.
        device: Torch execution device.
        
    Returns:
        Dict containing:
            'probability_map': [H, W] float32
            'binary_mask': [H, W] uint8 (0 or 1)
            'slick_pixels': int
            'inference_time_sec': float
            'patches_processed': int
    """
    t0 = time.time()
    h, w, c = sar_image.shape
    assert c == 2, f"Expected 2-channel SAR image (VH, VV), got {c} channels."

    norm_sar = preprocess_sar_imagery(sar_image)

    # Determine execution device from model if not passed
    if device is None:
        device = next(model.parameters()).device

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Generate patch coordinates
    y_coords = list(range(0, h - patch_size + 1, stride))
    if (h - patch_size) % stride != 0:
        y_coords.append(h - patch_size)

    x_coords = list(range(0, w - patch_size + 1, stride))
    if (w - patch_size) % stride != 0:
        x_coords.append(w - patch_size)

    patches = []
    positions = []

    for y in y_coords:
        for x in x_coords:
            patch = norm_sar[y:y+patch_size, x:x+patch_size, :]
            # Transpose to [C, H, W]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1))
            patches.append(patch_tensor)
            positions.append((y, x))

    total_patches = len(patches)

    with torch.no_grad():
        for i in range(0, total_patches, batch_size):
            batch_patches = torch.stack(patches[i:i+batch_size]).to(device)
            logits = model(batch_patches)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for j in range(len(probs)):
                y, x = positions[i + j]
                prob_map[y:y+patch_size, x:x+patch_size] += probs[j]
                count_map[y:y+patch_size, x:x+patch_size] += 1.0

    # Average overlapping areas
    count_map = np.maximum(count_map, 1.0)
    prob_map /= count_map

    binary_mask = (prob_map >= threshold).astype(np.uint8)
    slick_pixels = int(np.sum(binary_mask > 0))
    elapsed = time.time() - t0

    return {
        "probability_map": prob_map,
        "binary_mask": binary_mask,
        "slick_pixels": slick_pixels,
        "total_pixels": h * w,
        "slick_ratio": slick_pixels / (h * w),
        "inference_time_sec": elapsed,
        "patches_processed": total_patches,
        "threshold": threshold,
    }
