"""
Local Verification Script for Code Correctness
----------------------------------------------
Tests imports, model construction, 1-channel tensor forwarding,
and loss calculation without heavy GPU compute.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from src.models import build_segformer_b0, verify_model_forward
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetricsMeter

print("Testing code integrity...", flush=True)

# 1. Config loading
with open("configs/exp1_segformer_b0.yaml", "r") as f:
    cfg = yaml.safe_load(f)
print("  [PASS] Config YAML parsed successfully.", flush=True)

# 2. Model builder & 1-channel adapter
model = build_segformer_b0(in_channels=1, classes=1, encoder_weights=None, verbose=False)
print("  [PASS] smp.Segformer mit_b0 instantiated.", flush=True)

# 3. Model forward pass
device = torch.device("cpu")
verify_model_forward(model, device, input_size=256)
print("  [PASS] 1-Channel forward pass verified.", flush=True)

# 4. Loss calculation
criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
dummy_logits = torch.randn(2, 1, 256, 256)
dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
loss, breakdown = criterion(dummy_logits, dummy_targets)
assert not torch.isnan(loss), "Loss returned NaN!"
print(f"  [PASS] Loss computed successfully: total={loss.item():.4f}, bce={breakdown['bce_loss']:.4f}, dice={breakdown['dice_loss']:.4f}", flush=True)

# 5. Metrics computation
meter = SegmentationMetricsMeter(threshold=0.5)
meter.update(dummy_logits, dummy_targets)
metrics = meter.compute()
print(f"  [PASS] Metrics computed: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}", flush=True)

print("\nALL LOCAL COMPONENT INTEGRITY CHECKS PASSED.", flush=True)
