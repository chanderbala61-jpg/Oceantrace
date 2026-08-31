"""
OceanTrace - Model Evaluation Module

Evaluates the trained U-Net checkpoint on the test dataset split, calculates
IoU, Dice, Precision, and Recall metrics, saves results to outputs/metrics/,
and generates 10 visual evaluation comparisons under outputs/visualizations/.
"""

import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import OilSpillDataset
from src.model import UNet, compute_metrics


def evaluate_model(
    checkpoint_path: str = None,
    data_dir: str = 'data/raw',
    output_metrics_dir: str = 'outputs/metrics',
    output_vis_dir: str = 'outputs/visualizations',
    threshold: float = 0.5
):
    """
    Evaluates trained U-Net checkpoint on test split and generates visual outputs.
    """
    if checkpoint_path is None:
        fast_path = 'models/checkpoints/best_unet_model_fast.pth'
        std_path = 'models/checkpoints/best_unet_model.pth'
        checkpoint_path = fast_path if os.path.exists(fast_path) else std_path
    print("==================================================================")
    print("            OCEANTRACE MODEL EVALUATION PIPELINE                 ")
    print("==================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Evaluation Device: {device}")
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    
    # Load Model Checkpoint (supports both UNet and UNetFast)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get('architecture') == 'UNetFast' or checkpoint.get('base_channels') == 16:
        from src.train_fast import UNetFast
        model = UNetFast(in_channels=3, out_channels=1, base_channels=checkpoint.get('base_channels', 16)).to(device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"-> Successfully loaded checkpoint from {checkpoint_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load Test Dataset
    test_dataset = OilSpillDataset(
        mode='test',
        raw_data_dir=data_dir,
        use_augmentation=False
    )
    
    if len(test_dataset) == 0:
        print("[Warning] No test samples found. Falling back to validation set evaluation.")
        val_csv = os.path.join(data_dir, 'train', 'dataframe_val_dataset_256_90.csv')
        test_dataset = OilSpillDataset(
            mode='val',
            csv_path=val_csv if os.path.exists(val_csv) else None,
            raw_data_dir=data_dir,
            use_augmentation=False
        )
        
    print(f"-> Test Samples to Evaluate: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    sample_count = 0
    
    vis_count = 0
    max_vis = 10
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device) # [1, C, H, W]
            masks = batch['mask'].to(device)   # [1, 1, H, W]
            img_path = batch['image_path'][0]
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            metrics = compute_metrics(probs, masks, threshold=threshold)
            
            total_iou += metrics['iou']
            total_dice += metrics['dice']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            sample_count += 1
            
            # Generate Visual Comparisons for up to 10 samples
            if vis_count < max_vis:
                vis_count += 1
                
                img_np = images[0].cpu().numpy().transpose(1, 2, 0) # [H, W, C]
                img_uint8 = (img_np * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) if img_np.shape[2] == 3 else cv2.cvtColor(img_uint8[:, :, 0], cv2.COLOR_GRAY2BGR)
                
                gt_np = (masks[0, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_np = (preds[0, 0].cpu().numpy() * 255).astype(np.uint8)
                
                # Format 3-channel masks for display
                gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
                
                # Create Color Overlay (Red highlight for predicted spill)
                overlay_bgr = img_bgr.copy()
                red_mask = np.zeros_like(img_bgr)
                red_mask[:, :, 2] = pred_np # Red channel
                overlay_bgr = cv2.addWeighted(overlay_bgr, 0.7, red_mask, 0.5, 0)
                
                # Combine 4 Panels: [Original | Ground Truth | Prediction | Overlay]
                panel_h, panel_w = img_bgr.shape[:2]
                canvas = np.zeros((panel_h + 40, panel_w * 4 + 30, 3), dtype=np.uint8)
                canvas.fill(20) # dark navy background
                
                canvas[35:35+panel_h, 5:5+panel_w] = img_bgr
                canvas[35:35+panel_h, panel_w+10:panel_w*2+10] = gt_bgr
                canvas[35:35+panel_h, panel_w*2+15:panel_w*3+15] = pred_bgr
                canvas[35:35+panel_h, panel_w*3+20:panel_w*4+20] = overlay_bgr
                
                # Text Titles
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, "Original SAR", (10, 25), font, 0.5, (255, 255, 255), 1)
                cv2.putText(canvas, "Ground Truth Mask", (panel_w + 15, 25), font, 0.5, (0, 255, 255), 1)
                cv2.putText(canvas, "Predicted Mask", (panel_w * 2 + 20, 25), font, 0.5, (0, 255, 0), 1)
                cv2.putText(canvas, "Red Overlay", (panel_w * 3 + 25, 25), font, 0.5, (0, 0, 255), 1)
                
                vis_save_path = os.path.join(output_vis_dir, f"eval_sample_{vis_count}.png")
                cv2.imwrite(vis_save_path, canvas)
                print(f"  -> Generated Visual Comparison: {vis_save_path}")
                
    mean_iou = total_iou / sample_count if sample_count > 0 else 0.0
    mean_dice = total_dice / sample_count if sample_count > 0 else 0.0
    mean_precision = total_precision / sample_count if sample_count > 0 else 0.0
    mean_recall = total_recall / sample_count if sample_count > 0 else 0.0
    
    print("\n--- TEST EVALUATION METRICS SUMMARY ---")
    print(f"  - Mean IoU:       {mean_iou:.4f}")
    print(f"  - Mean Dice:      {mean_dice:.4f}")
    print(f"  - Mean Precision: {mean_precision:.4f}")
    print(f"  - Mean Recall:    {mean_recall:.4f}")
    
    metrics_summary = {
        'test_samples_evaluated': sample_count,
        'threshold': threshold,
        'mean_iou': round(mean_iou, 4),
        'mean_dice': round(mean_dice, 4),
        'mean_precision': round(mean_precision, 4),
        'mean_recall': round(mean_recall, 4)
    }
    
    metrics_save_path = os.path.join(output_metrics_dir, 'test_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"-> Test metrics report saved to: {metrics_save_path}")
    return metrics_summary


if __name__ == '__main__':
    evaluate_model()
