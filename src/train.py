"""
OceanTrace - Training Pipeline

Executes U-Net model training, validation loops, loss evaluation, metric tracking,
and saves the best model checkpoint to models/checkpoints/best_unet_model.pth.
"""

import os
import sys
import json
import argparse
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

from src.dataset import OilSpillDataset
from src.model import UNet, BCEDiceLoss, compute_metrics


def train_model(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    data_dir: str = 'data/raw',
    checkpoint_dir: str = 'models/checkpoints',
    device_name: str = None,
    max_train_samples: int = None,
    max_val_samples: int = None
):
    """
    Main training routine for OceanTrace U-Net model.
    """
    print("==================================================================")
    print("            OCEANTRACE U-NET TRAINING PIPELINE                   ")
    print("==================================================================")

    # Determine execution device (CUDA GPU if available, else CPU)
    if device_name is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
        
    print(f"-> Execution Device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_unet_model.pth')
    
    # Dataset CSV paths
    train_csv = os.path.join(data_dir, 'train', 'dataframe_train_dataset_256_90.csv')
    val_csv = os.path.join(data_dir, 'train', 'dataframe_val_dataset_256_90.csv')
    
    print("-> Loading Training Dataset...")
    train_dataset = OilSpillDataset(
        mode='train',
        csv_path=train_csv if os.path.exists(train_csv) else None,
        raw_data_dir=data_dir,
        use_augmentation=True
    )
    if max_train_samples and max_train_samples < len(train_dataset):
        # Subset with stratified balance if available
        import random
        random.seed(42)
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        train_dataset.samples = [train_dataset.samples[i] for i in indices[:max_train_samples]]
    
    print("-> Loading Validation Dataset...")
    val_dataset = OilSpillDataset(
        mode='val',
        csv_path=val_csv if os.path.exists(val_csv) else None,
        raw_data_dir=data_dir,
        use_augmentation=False
    )
    if max_val_samples and max_val_samples < len(val_dataset):
        val_dataset.samples = val_dataset.samples[:max_val_samples]
    
    print(f"-> Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize Model, Loss Function, and Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch}/{epochs}] Batch {batch_idx + 1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}", flush=True)
            
        epoch_train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
        
        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                running_val_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(logits)
                metrics = compute_metrics(probs, masks, threshold=0.5)
                
                total_iou += metrics['iou']
                total_dice += metrics['dice']
                val_batches += 1
                
        epoch_val_loss = running_val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        epoch_val_iou = total_iou / val_batches if val_batches > 0 else 0.0
        epoch_val_dice = total_dice / val_batches if val_batches > 0 else 0.0
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)
        history['val_iou'].append(epoch_val_iou)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Dice: {epoch_val_dice:.4f} | Val IoU: {epoch_val_iou:.4f}")
        
        # Save Best Checkpoint
        if epoch_val_dice >= best_val_dice or epoch == 1:
            best_val_dice = epoch_val_dice
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': epoch_val_dice,
                'val_iou': epoch_val_iou,
                'val_loss': epoch_val_loss
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  --> Saved Best Checkpoint: {best_checkpoint_path} (Dice: {epoch_val_dice:.4f})")
            
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds. Best Val Dice: {best_val_dice:.4f}")
    
    # Save History Metadata
    history_path = os.path.join(checkpoint_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    return best_checkpoint_path, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OceanTrace U-Net Segmentation Model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Path to raw data folder')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Limit training samples for quick runs')
    parser.add_argument('--max-val-samples', type=int, default=None, help='Limit validation samples for quick runs')
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
