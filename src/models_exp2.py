"""
Experiment 2 Model Builder Module
---------------------------------
Constructs 2-channel SegFormer-B0 (MiT-B0 encoder + All-MLP decoder)
supporting dual-polarization SAR inputs: [B, 2, 256, 256] -> [B, 1, 256, 256]
without RGB duplication or channel dropping.
"""

import torch
import torch.nn as nn
from typing import Optional
import segmentation_models_pytorch as smp


def build_exp2_segformer_b0(
    in_channels: int = 2,
    classes: int = 1,
    encoder_weights: Optional[str] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Builds SegFormer-B0 using segmentation_models_pytorch with 2-channel SAR input (VH + VV).
    
    Args:
        in_channels: Number of input channels (2 for dual-pol VH + VV).
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
        print("MODEL SUMMARY: Experiment 2 SegFormer-B0 (2-Channel)")
        print("==================================================")
        print(f"  - Encoder: mit_b0 (Pretrained: {encoder_weights})")
        print(f"  - Input channels: {in_channels} (Dual-polarization SAR: VH + VV)")
        print(f"  - Output classes: {classes} (Binary Oil Spill)")
        print(f"  - First conv shape: {model.encoder.patch_embed1.proj.weight.shape}")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
        print("==================================================")

    return model
