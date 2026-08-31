"""
OceanTrace - Fast CPU-Optimised Training Pipeline
==================================================

Hackathon-optimised variant of the training pipeline.

KEY DIFFERENCES from train.py
  - Stride=256 (non-overlapping patches): ~2,700 train samples vs 21,744
  - Batches/epoch: ~338 vs 1,359 (with batch_size=8)
  - Lighter UNetFast (base_channels=16, ~0.5M params vs 2M)
  - Early stopping (patience=3)
  - ReduceLROnPlateau scheduler
  - --smoke-test flag: 200-sample subset to verify full pipeline end-to-end
  - --inspect-only flag: print dataset stats and exit
  - Progress bars with per-batch ETA
  - num_workers=0 (Windows embedded Python compatibility)
  - pin_memory=False (CPU-only build)

Does NOT fabricate or alter metrics. All reported numbers come from
real inference on real validation samples.

Usage
-----
# Pipeline verification (200 train / 60 val patches):
python src/train_fast.py --smoke-test

# Full hackathon training:
python src/train_fast.py --epochs 5 --batch-size 8

# Dataset stats only:
python src/train_fast.py --inspect-only
"""

import os
import sys
import json
import argparse
import time
import random
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.dataset import OilSpillDataset
from src.model import BCEDiceLoss, compute_metrics


# ---------------------------------------------------------------------------
# Lightweight U-Net (base_channels=16 => ~0.5M params, ~2x faster on CPU)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetFast(nn.Module):
    """
    Lightweight U-Net with configurable base channel width.
      base_channels=16  =>  ~0.5 M params  (fast CPU training)
      base_channels=32  =>  ~2.0 M params  (matches original model.py UNet)

    Checkpoint saved by this script includes 'architecture': 'UNetFast' and
    'base_channels' key so the dashboard can reload it correctly.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1 = _DoubleConv(in_channels, b)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b,   b*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b*2, b*4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b*4, b*8))
        self.bot  = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b*8, b*8))

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc1  = _DoubleConv(b*16, b*4)
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc2  = _DoubleConv(b*8,  b*2)
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc3  = _DoubleConv(b*4,  b)
        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc4  = _DoubleConv(b*2,  b)
        self.out  = nn.Conv2d(b, out_channels, kernel_size=1)

    @staticmethod
    def _cat(up_feat, skip):
        dy = skip.size(2) - up_feat.size(2)
        dx = skip.size(3) - up_feat.size(3)
        up_feat = F.pad(up_feat, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        return torch.cat([skip, up_feat], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)
        x  = self.dc1(self._cat(self.up1(b),  e4))
        x  = self.dc2(self._cat(self.up2(x),  e3))
        x  = self.dc3(self._cat(self.up3(x),  e2))
        x  = self.dc4(self._cat(self.up4(x),  e1))
        return self.out(x)


# ---------------------------------------------------------------------------
# Non-overlapping patch dataset (stride=256, no data fabrication)
# ---------------------------------------------------------------------------

class NonOverlapDataset(Dataset):
    """
    Reads from the SAME SAR scenes referenced in the CSV and samples patches
    using non-overlapping stride (or stride matching patch size) while preserving
    the exact train/val split.
    """
    def __init__(self, base_dataset: OilSpillDataset, stride: int = 256, max_samples: int = None, seed: int = 42):
        self.base = base_dataset
        pH, pW = base_dataset.patch_size  # (256, 256)
        
        # If base dataset has samples from CSV, filter samples where (coord_y % stride == 0 and coord_x % stride == 0)
        # or generate grid coordinates per scene
        self.samples = []
        if len(base_dataset.samples) > 0 and base_dataset.samples[0]['coord_y'] is not None:
            # Filter existing CSV patches to non-overlapping subset to preserve identical distribution
            seen_coords = set()
            for s in base_dataset.samples:
                cy, cx = s['coord_y'], s['coord_x']
                # Pick grid points spaced by at least stride
                grid_y = (cy // stride) * stride
                grid_x = (cx // stride) * stride
                key = (s['image_path'], grid_y, grid_x)
                if key not in seen_coords:
                    seen_coords.add(key)
                    self.samples.append(s)
        else:
            self.samples = list(base_dataset.samples)

        if max_samples and max_samples < len(self.samples):
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        saved = self.base.samples
        self.base.samples = [self.samples[idx]]
        item = self.base[0]
        self.base.samples = saved
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fast(
    epochs=5,
    batch_size=8,
    learning_rate=1e-3,
    data_dir='data/raw',
    checkpoint_dir='models/checkpoints',
    base_channels=16,
    smoke_test=False,
    patience=3,
    inspect_only=False,
):
    print("=" * 68)
    label = "[SMOKE TEST] Pipeline Verification" if smoke_test else "Hackathon Fast Training"
    print(f"  OceanTrace UNetFast — {label}")
    print("=" * 68)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device     : {device.type.upper()}")
    if device.type == 'cuda':
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    else:
        import multiprocessing
        print(f"  CPU cores  : {multiprocessing.cpu_count()}")
    print(f"  PyTorch    : {torch.__version__}")

    train_csv = os.path.join(data_dir, 'train', 'dataframe_train_dataset_256_90.csv')
    val_csv   = os.path.join(data_dir, 'train', 'dataframe_val_dataset_256_90.csv')

    print("\n  Loading base datasets...")
    base_train = OilSpillDataset(
        mode='train',
        csv_path=train_csv if os.path.exists(train_csv) else None,
        raw_data_dir=data_dir,
        use_augmentation=True,
    )
    base_val = OilSpillDataset(
        mode='val',
        csv_path=val_csv if os.path.exists(val_csv) else None,
        raw_data_dir=data_dir,
        use_augmentation=False,
    )

    STRIDE      = 256
    SMOKE_TRAIN = 200
    SMOKE_VAL   = 60

    print(f"  Rebuilding with non-overlapping stride={STRIDE}...")
    train_dataset = NonOverlapDataset(
        base_train, stride=STRIDE,
        max_samples=SMOKE_TRAIN if smoke_test else None, seed=42
    )
    val_dataset = NonOverlapDataset(
        base_val, stride=STRIDE,
        max_samples=SMOKE_VAL if smoke_test else None, seed=42
    )

    n_train          = len(train_dataset)
    n_val            = len(val_dataset)
    batches_per_epoch = math.ceil(n_train / batch_size)

    # Timing estimate: ~0.40s/batch for base=16, bs=8 on 8-core CPU
    secs_per_batch  = 0.40 if base_channels == 16 else 0.90
    est_epoch_s     = batches_per_epoch * secs_per_batch
    est_total_s     = est_epoch_s * epochs * 1.15

    print()
    print("=" * 68)
    print("  DATASET SIZE       : 14 train scenes + 7 test scenes (SAR .tif)")
    print(f"  TRAINING SAMPLES   : {n_train:,}{' [SMOKE TEST SUBSET]' if smoke_test else ''}")
    print(f"  VAL SAMPLES        : {n_val:,}{' [SMOKE TEST SUBSET]' if smoke_test else ''}")
    print(f"  BATCH SIZE         : {batch_size}")
    print(f"  BATCHES/EPOCH      : {batches_per_epoch}  (was 1,359 with stride=90, bs=16)")
    print(f"  IMAGE SIZE         : 256x256 patches from large SAR scenes")
    print(f"  DEVICE             : {device.type.upper()} — CUDA unavailable (CPU-only build)")
    print(f"  DATALOADER         : num_workers=0, pin_memory=False (Windows CPU)")
    print(f"  U-NET PARAMS       : {base_channels} base channels — ", end="")
    print("~0.5M params (FAST)" if base_channels == 16 else "~2M params (STANDARD)")
    print(f"  ESTIMATED/EPOCH    : ~{fmt_time(est_epoch_s)}")
    print(f"  PROPOSED EPOCHS    : {epochs}  (early-stop patience={patience})")
    print(f"  ESTIMATED TOTAL    : ~{fmt_time(est_total_s)}")
    print("=" * 68)

    if inspect_only:
        print("\n  --inspect-only: exiting without training.\n")
        return None, {}

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  num_workers=0, pin_memory=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    model     = UNetFast(in_channels=3, out_channels=1,
                         base_channels=base_channels).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print(f"\n  Model parameters   : {count_parameters(model):,}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path     = os.path.join(checkpoint_dir, 'best_unet_model_fast.pth')
    best_val_dice = 0.0
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': [],
                     'val_dice':   [], 'val_iou':  [],
                     'epoch_time_s': []}

    global_start = time.time()
    print(f"\n  Training started — {epochs} epochs\n" + "-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        for bi, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            if (bi + 1) % 50 == 0 or (bi + 1) == len(train_loader):
                elapsed  = time.time() - t0
                eta      = (elapsed / (bi + 1)) * (len(train_loader) - bi - 1)
                print(f"  [Ep {epoch:02d}/{epochs}] "
                      f"Batch {bi+1:>4}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | ETA: {fmt_time(eta)}",
                      flush=True)

        tr_loss = running_loss / n_train if n_train > 0 else 0.0

        # --- Validate ---
        model.eval()
        vl_loss = vl_dice = vl_iou = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks  = batch['mask'].to(device)
                logits = model(images)
                vl_loss += criterion(logits, masks).item() * images.size(0)
                m = compute_metrics(torch.sigmoid(logits), masks, threshold=0.5)
                vl_dice += m['dice']
                vl_iou  += m['iou']
                vb += 1

        val_loss = vl_loss / n_val if n_val > 0 else 0.0
        val_dice = vl_dice / vb   if vb    > 0 else 0.0
        val_iou  = vl_iou  / vb   if vb    > 0 else 0.0
        ep_time  = time.time() - t0

        scheduler.step(val_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['epoch_time_s'].append(round(ep_time, 1))

        print(f"\n  Epoch [{epoch:02d}/{epochs}] "
              f"| Train: {tr_loss:.4f} "
              f"| Val: {val_loss:.4f} "
              f"| Dice: {val_dice:.4f} "
              f"| IoU: {val_iou:.4f} "
              f"| {fmt_time(ep_time)}")

        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            no_improve    = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice':      val_dice,
                'val_iou':       val_iou,
                'val_loss':      val_loss,
                'base_channels': base_channels,
                'architecture':  'UNetFast',
                'smoke_test':    smoke_test,
            }, ckpt_path)
            print(f"  --> Best checkpoint saved (Dice: {val_dice:.4f})")
        else:
            no_improve += 1
            print(f"  --> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n  [Early Stop] Stopping after {patience} epochs without improvement.")
                break

        print("-" * 68)

    total_t = time.time() - global_start
    print(f"\n{'=' * 68}")
    if smoke_test:
        print("  [SMOKE TEST PASSED] Full pipeline works end-to-end.")
        print("  NOTE: Metrics above are from a 200-sample subset — not representative.")
        print("  Run without --smoke-test for real training.")
    else:
        print("  TRAINING COMPLETE")
    print(f"  Total time   : {fmt_time(total_t)}")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint   : {ckpt_path}")
    print("=" * 68)

    hist_path = os.path.join(checkpoint_dir, 'train_history_fast.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History      : {hist_path}\n")

    return ckpt_path, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OceanTrace — Fast CPU-Optimised U-Net Training'
    )
    parser.add_argument('--epochs',         type=int,   default=5)
    parser.add_argument('--batch-size',     type=int,   default=8)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--data-dir',       type=str,   default='data/raw')
    parser.add_argument('--checkpoint-dir', type=str,   default='models/checkpoints')
    parser.add_argument('--base-channels',  type=int,   default=16, choices=[16, 32])
    parser.add_argument('--smoke-test',     action='store_true',
                        help='Run on 200-sample subset to verify pipeline end-to-end')
    parser.add_argument('--patience',       type=int,   default=3)
    parser.add_argument('--inspect-only',   action='store_true',
                        help='Print dataset stats and exit without training')
    args = parser.parse_args()

    train_fast(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        base_channels=args.base_channels,
        smoke_test=args.smoke_test,
        patience=args.patience,
        inspect_only=args.inspect_only,
    )
