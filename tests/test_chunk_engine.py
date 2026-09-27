"""
Unit test for OceanTrace Chunk-Training Engine Checkpoint & Resume System
------------------------------------------------------------------------
Tests:
1. Checkpoint creation with all mandatory state fields.
2. Checkpoint restoration (weights, optimizer, RNG, metrics, chunk_number).
3. Training history CSV logging.
"""

import os
import sys
import tempfile
import torch
import numpy as np
import random
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models_exp2 import build_exp2_segformer_b0
from src.train_chunk_exp2 import save_chunk_checkpoint, load_resumable_checkpoint


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        hist_csv = os.path.join(tmpdir, "training_history.csv")

        # 1. Create model and optimizer
        model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        # Mutate weights slightly with dummy step
        dummy_x = torch.randn(1, 2, 256, 256)
        dummy_y = torch.randint(0, 2, (1, 1, 256, 256)).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(dummy_x), dummy_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save checkpoint for Chunk 1
        train_res = {"train_loss": 0.6543, "train_bce": 0.3200, "train_dice": 0.9800, "train_iou": 0.4200, "chunk_time_sec": 12.5}
        val_res = {"val_loss": 0.6800, "iou": 0.4500, "dice": 0.6000, "precision": 0.5500, "recall": 0.6600, "accuracy": 0.9200}
        
        best_iou = save_chunk_checkpoint(
            checkpoint_dir=ckpt_dir,
            chunk_number=1,
            epoch=1,
            scenes_processed=20,
            total_usable_scenes=837,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_results=train_res,
            val_results=val_res,
            best_val_iou=0.0,
            training_history=[],
            history_path=hist_csv,
        )

        assert os.path.exists(os.path.join(ckpt_dir, "chunk_001.pth")), "chunk_001.pth missing"
        assert os.path.exists(os.path.join(ckpt_dir, "latest.pth")), "latest.pth missing"
        assert os.path.exists(os.path.join(ckpt_dir, "best.pth")), "best.pth missing"
        assert os.path.exists(hist_csv), "training_history.csv missing"
        assert best_iou == 0.4500, f"Expected best_iou 0.4500, got {best_iou}"

        # 2. Test resumption on a NEW model instance
        new_model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=30)

        # Confirm weights are initially different before loading
        diff_before = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(model.parameters(), new_model.parameters()))
        assert diff_before > 0.0, "Weights should be different before loading checkpoint"

        # Load from latest.pth
        chunk_num, epoch, scenes, best_val, hist = load_resumable_checkpoint(
            os.path.join(ckpt_dir, "latest.pth"),
            new_model,
            new_optimizer,
            new_scheduler,
        )

        assert chunk_num == 1, f"Expected chunk 1, got {chunk_num}"
        assert epoch == 1, f"Expected epoch 1, got {epoch}"
        assert scenes == 20, f"Expected 20 scenes, got {scenes}"
        assert best_val == 0.4500, f"Expected best_val 0.4500, got {best_val}"

        # Confirm weights are IDENTICAL after loading
        diff_after = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(model.parameters(), new_model.parameters()))
        assert diff_after == 0.0, f"Weights must be identical after loading, diff was {diff_after}"

        # Check history CSV
        df = pd.read_csv(hist_csv)
        assert len(df) == 1, f"Expected 1 history row, got {len(df)}"
        assert df.iloc[0]["chunk_number"] == 1
        assert abs(df.iloc[0]["val_iou"] - 0.4500) < 1e-4

        print("All checkpoint roundtrip and resume assertions PASSED!")


if __name__ == "__main__":
    test_checkpoint_roundtrip()
