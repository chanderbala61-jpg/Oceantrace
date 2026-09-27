"""
Loss Functions for Experiment 1: Dice + BCE Loss
------------------------------------------------
Computes combined 0.5 * Dice Loss + 0.5 * BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Operates directly on raw logits using sigmoid.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten batch and spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice_coeff


class DiceBCELoss(nn.Module):
    """
    Combined 0.5 * Dice Loss + 0.5 * BCEWithLogitsLoss.
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_fn = DiceLoss(smooth=smooth)
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = self.bce_fn(logits, targets)
        dice = self.dice_fn(logits, targets)
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        breakdown = {
            "loss": total_loss.item(),
            "bce_loss": bce.item(),
            "dice_loss": dice.item()
        }
        return total_loss, breakdown
