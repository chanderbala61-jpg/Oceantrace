"""
OceanTrace - Single Image & Batch Prediction Engine

Loads trained U-Net checkpoint, executes inference on satellite SAR image,
generates probability map, binary spill mask, red overlay visualization,
and estimates detected spill pixels and physical area.
"""

import os
import argparse
import torch
import cv2
import numpy as np

from src.preprocessing import apply_speckle_filter, normalize_image
from src.model import UNet


def predict_image(
    image_path: str,
    checkpoint_path: str = 'models/checkpoints/best_unet_model.pth',
    output_dir: str = 'outputs/predictions',
    threshold: float = 0.5,
    pixel_area_sq_m: float = None # e.g. 100.0 m^2 per pixel if metadata available
):
    """
    Executes inference pipeline for a single satellite image file.
    
    Returns:
        dict: {
            'image': np.ndarray,
            'probability_map': np.ndarray,
            'spill_mask': np.ndarray,
            'overlay': np.ndarray,
            'detected_pixels': int,
            'estimated_area_sq_m': float or None,
            'output_overlay_path': str
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found: {image_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Image
    try:
        from PIL import Image
        with Image.open(image_path) as pil_img:
            image_rgb = np.array(pil_img)
            if image_rgb.ndim == 2:
                image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=-1)
            elif image_rgb.shape[-1] == 4:
                image_rgb = image_rgb[:, :, :3]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if image_rgb.dtype == np.uint8 else (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    except Exception:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not decode image at path: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
    H, W = image_rgb.shape[:2]
    
    # Preprocessing
    filtered_img = apply_speckle_filter(image_rgb, method='median', kernel_size=3)
    norm_img = normalize_image(filtered_img, method='minmax')
    
    # Resize to 256x256 for model input
    input_resized = cv2.resize(norm_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    input_tensor = torch.from_numpy(input_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Load Model (supports both UNet and UNetFast)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get('architecture') == 'UNetFast' or checkpoint.get('base_channels') == 16:
            from src.train_fast import UNetFast
            model = UNetFast(in_channels=3, out_channels=1, base_channels=checkpoint.get('base_channels', 16)).to(device)
        else:
            model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running initialized model.")
        model = UNet(in_channels=3, out_channels=1).to(device)
        
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs_resized = torch.sigmoid(logits)[0, 0].cpu().numpy()
        
    # Resize probability map back to original input dimensions
    prob_map = cv2.resize(probs_resized, (W, H), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    
    # Calculate spill pixels
    detected_pixels = int(np.sum(binary_mask))
    estimated_area = (detected_pixels * pixel_area_sq_m) if pixel_area_sq_m is not None else None
    
    # Generate Overlay
    red_mask = np.zeros_like(image_bgr)
    red_mask[:, :, 2] = binary_mask * 255
    overlay_bgr = cv2.addWeighted(image_bgr, 0.7, red_mask, 0.5, 0)
    
    # Save Prediction Artifacts
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    overlay_save_path = os.path.join(output_dir, f"{base_name}_prediction.png")
    mask_save_path = os.path.join(output_dir, f"{base_name}_mask.png")
    
    cv2.imwrite(overlay_save_path, overlay_bgr)
    cv2.imwrite(mask_save_path, binary_mask * 255)
    
    print(f"-> Prediction Completed for {base_name}")
    print(f"  - Detected Spill Pixels: {detected_pixels}")
    print(f"  - Overlay Saved to: {overlay_save_path}")
    
    return {
        'image': image_rgb,
        'probability_map': prob_map,
        'spill_mask': binary_mask,
        'overlay': cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        'detected_pixels': detected_pixels,
        'estimated_area_sq_m': estimated_area,
        'output_overlay_path': overlay_save_path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OceanTrace Oil Spill Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to satellite image')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_unet_model.pth', help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold [0.0 - 1.0]')
    args = parser.parse_args()

    predict_image(image_path=args.image, checkpoint_path=args.checkpoint, threshold=args.threshold)
