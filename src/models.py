"""
SegFormer Model Builder for Sentinel-1 SAR Oil Spill Segmentation
-----------------------------------------------------------------
Constructs SegFormer-B0 (MiT-B0 encoder + lightweight All-MLP decoder)
with genuine 1-channel SAR input support.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import segmentation_models_pytorch as smp


def build_segformer_b0(
    in_channels: int = 1,
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
    verbose: bool = True
) -> nn.Module:
    """
    Builds SegFormer-B0 using segmentation_models_pytorch with 1-channel SAR input.
    
    Args:
        in_channels: Number of input channels (1 for Sentinel-1 VV SAR).
        classes: Number of output segmentation classes (1 for binary oil spill).
        encoder_weights: Pretrained weights ("imagenet" or None).
        verbose: If True, prints parameter statistics.
        
    Returns:
        torch.nn.Module SegFormer model.
    """
    try:
        model = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate smp.Segformer with in_channels={in_channels}, "
            f"encoder_weights='{encoder_weights}'. Error: {e}"
        )

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("==================================================")
        print("MODEL SUMMARY: SegFormer-B0 (MiT-B0 + All-MLP)")
        print("==================================================")
        print(f"  - Encoder: mit_b0 (Pretrained: {encoder_weights})")
        print(f"  - Input channels: {in_channels} (Single-band SAR VV dB)")
        print(f"  - Output classes: {classes} (Binary Oil Spill)")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
        print("==================================================")

    return model


def verify_model_forward(model: nn.Module, device: torch.device, input_size: int = 256) -> bool:
    """
    Verifies that the SegFormer model accepts (B, 1, H, W) tensor and produces (B, 1, H, W) logits.
    """
    model.eval()
    dummy_input = torch.randn(2, 1, input_size, input_size, device=device)
    with torch.no_grad():
        out = model(dummy_input)
    
    expected_shape = (2, 1, input_size, input_size)
    if out.shape != expected_shape:
        raise ValueError(f"Output shape mismatch! Expected {expected_shape}, got {out.shape}")
    
    return True
