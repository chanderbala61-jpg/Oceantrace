"""
Metrics Suite for Binary Oil Spill Segmentation
-----------------------------------------------
Calculates IoU (Jaccard), Dice (F1), Precision, Recall, Pixel Accuracy,
and raw confusion matrix counts (TP, TN, FP, FN).
"""

from typing import Dict
import torch
import numpy as np


class SegmentationMetricsMeter:
    """
    Accumulates pixel-level confusion matrix counts across batches
    and calculates final global metrics.
    """
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Updates counts from a batch of logits and binary targets.
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).long()
            targets = (targets >= 0.5).long()

            preds_flat = preds.view(-1)
            targets_flat = targets.view(-1)

            tp = (preds_flat * targets_flat).sum().item()
            fp = (preds_flat * (1 - targets_flat)).sum().item()
            fn = ((1 - preds_flat) * targets_flat).sum().item()
            tn = ((1 - preds_flat) * (1 - targets_flat)).sum().item()

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn

    def compute(self) -> Dict[str, float]:
        """
        Computes global accumulated metrics.
        """
        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        tn = float(self.tn)

        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        dice = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        f1 = dice  # For binary classification, Dice is equivalent to F1
        accuracy = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)

        return {
            "iou": float(iou),
            "dice": float(dice),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "threshold": self.threshold
        }
