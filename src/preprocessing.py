"""
OceanTrace - Preprocessing Module

Provides modular preprocessing utilities for SAR satellite imagery and oil spill masks:
1. Configurable SAR Speckle Noise Filtering (Median Filter / Lee Filter)
2. Per-Image Min-Max Normalization to [0, 1]
3. 256x256 Patch Extraction & Dataset CSV Coordinate Resolution
4. Categorical/Binary Mask Preservation (Nearest-Neighbor Scaling)
"""

import os
import cv2
import numpy as np
import pandas as pd


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies median filtering to reduce SAR speckle noise while preserving crisp edges.
    
    Args:
        image (np.ndarray): Input SAR image (H, W) or (H, W, C).
        kernel_size (int): Size of median filter window (must be odd integer >= 3).
        
    Returns:
        np.ndarray: Filtered SAR image.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        kernel_size = 3
        
    # Ensure float32 / uint8 compatibility without destroying dynamic range
    if image.dtype == np.float32 or image.dtype == np.float64:
        min_v, max_v = float(np.min(image)), float(np.max(image))
        rng = max_v - min_v
        if rng > 1e-6:
            norm_uint8 = ((image - min_v) / rng * 255).astype(np.uint8)
            filtered = cv2.medianBlur(norm_uint8, kernel_size)
            return (filtered.astype(np.float32) / 255.0) * rng + min_v
        else:
            return image
    else:
        return cv2.medianBlur(image, kernel_size)


def apply_lee_filter(image: np.ndarray, window_size: int = 5, cu: float = 0.25) -> np.ndarray:
    """
    Applies Lee filter for SAR multiplicative speckle noise reduction.
    
    Lee Filter Equation: R(x,y) = I_bar + W(x,y) * (I(x,y) - I_bar)
    where W(x,y) = Var(x,y) / (Var(x,y) + Var_noise)
    
    Args:
        image (np.ndarray): Input SAR image array.
        window_size (int): Local window size (odd integer).
        cu (float): Estimated noise coefficient of variation.
        
    Returns:
        np.ndarray: Speckle-filtered image array.
    """
    img_float = image.astype(np.float32)
    
    # Calculate local mean and local variance
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    local_sqr_mean = cv2.filter2D(img_float ** 2, -1, kernel)
    local_var = np.maximum(0.0, local_sqr_mean - (local_mean ** 2))
    
    # Estimate noise variance
    noise_var = (cu ** 2) * (local_mean ** 2)
    
    # Weight matrix W
    weights = local_var / (local_var + noise_var + 1e-8)
    weights = np.clip(weights, 0.0, 1.0)
    
    # Filtered response
    filtered = local_mean + weights * (img_float - local_mean)
    return filtered.astype(image.dtype)


def apply_speckle_filter(image: np.ndarray, method: str = 'median', **kwargs) -> np.ndarray:
    """
    Unified entry point for SAR noise filtering methods.
    
    Args:
        image (np.ndarray): Input image array.
        method (str): Filtering method ('median', 'lee', or 'none'/None).
        **kwargs: Method-specific parameters (kernel_size, window_size, etc.).
        
    Returns:
        np.ndarray: Filtered image array.
    """
    if method is None or method.lower() in ('none', 'off'):
        return image
    elif method.lower() == 'median':
        kernel_size = kwargs.get('kernel_size', 3)
        return apply_median_filter(image, kernel_size=kernel_size)
    elif method.lower() == 'lee':
        window_size = kwargs.get('window_size', 5)
        cu = kwargs.get('cu', 0.25)
        return apply_lee_filter(image, window_size=window_size, cu=cu)
    else:
        raise ValueError(f"Unsupported noise filter method: {method}. Choose 'median', 'lee', or 'none'.")


def normalize_image(image: np.ndarray, method: str = 'minmax', clip_percentiles: tuple = None) -> np.ndarray:
    """
    Performs per-image normalization to scale pixel values into [0.0, 1.0].
    
    Args:
        image (np.ndarray): Input image array.
        method (str): Normalization method ('minmax', 'zscore', or 'none').
        clip_percentiles (tuple): Optional (min_p, max_p) percentiles for robust scaling.
        
    Returns:
        np.ndarray: Normalized image array float32 in range [0.0, 1.0].
    """
    img_float = image.astype(np.float32)
    
    if method is None or method.lower() in ('none', 'off'):
        return img_float
        
    if clip_percentiles is not None:
        p_min, p_max = np.percentile(img_float, clip_percentiles)
        img_float = np.clip(img_float, p_min, p_max)
        
    if method.lower() == 'minmax':
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        range_val = max_val - min_val
        if range_val > 1e-8:
            normalized = (img_float - min_val) / range_val
        else:
            normalized = np.zeros_like(img_float, dtype=np.float32)
        return normalized.astype(np.float32)
        
    elif method.lower() == 'zscore':
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)
        if std_val > 1e-8:
            normalized = (img_float - mean_val) / std_val
        else:
            normalized = np.zeros_like(img_float, dtype=np.float32)
        return normalized.astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization method: {method}. Choose 'minmax', 'zscore', or 'none'.")


def process_mask(mask: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Ensures ground-truth segmentation masks remain binary categorical integer arrays.
    Never normalizes or applies float scaling/blurring to masks.
    
    Args:
        mask (np.ndarray): Input target mask array.
        target_size (tuple): Optional (width, height) to resize using NEAREST neighbor interpolation.
        
    Returns:
        np.ndarray: Binary mask array (uint8 with values 0 or 1).
    """
    # Convert multi-channel or float masks to uint8 binary (0 or 1)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
        
    binary_mask = (mask > 0).astype(np.uint8)
    
    if target_size is not None and (mask.shape[1], mask.shape[0]) != target_size:
        binary_mask = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
    return binary_mask


def extract_patches(image: np.ndarray, mask: np.ndarray = None, patch_size: tuple = (256, 256), stride: int = 90):
    """
    Extracts 256x256 image (and optional mask) patches using specified sliding window stride.
    
    Args:
        image (np.ndarray): Full input scene image (H, W) or (H, W, C).
        mask (np.ndarray): Optional full target mask (H, W).
        patch_size (tuple): (patch_height, patch_width).
        stride (int): Sliding window stride in pixels (default 90).
        
    Returns:
        list of dict: List of dicts containing patch images, masks, top-left coordinates, and scene metadata.
    """
    pH, pW = patch_size
    H, W = image.shape[:2]
    patches = []
    
    for y in range(0, H - pH + 1, stride):
        for x in range(0, W - pW + 1, stride):
            img_patch = image[y:y+pH, x:x+pW]
            mask_patch = mask[y:y+pH, x:x+pW] if mask is not None else None
            
            patches.append({
                'image': img_patch,
                'mask': mask_patch,
                'coord': (y, x),
                'has_oil': bool(np.any(mask_patch > 0)) if mask_patch is not None else False
            })
            
    return patches


def parse_dataset_csv(csv_path: str, raw_data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Reads dataset annotation CSV file (e.g., dataframe_train_dataset_256_90.csv),
    resolves paths to local files, parses top-left coordinates, and returns cleaned metadata.
    
    CSV Columns: paths, coordinates, class
    Example row: C:/Users/william/.../train\\images\\20200307.tif, "1576,1276", 1.0
    
    Args:
        csv_path (str): Path to annotation CSV file.
        raw_data_dir (str): Base path to raw data directory.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
                      ['image_path', 'mask_path', 'scene_name', 'coord_y', 'coord_x', 'class_label']
    """
    df = pd.read_csv(csv_path)
    records = []
    
    for _, row in df.iterrows():
        orig_path = str(row['paths'])
        # Extract filename (e.g., 20200307.tif)
        filename = os.path.basename(orig_path.replace('\\', '/'))
        scene_name = os.path.splitext(filename)[0]
        
        # Resolve local paths
        if 'test' in orig_path.lower():
            split_subfolder = 'test'
        else:
            split_subfolder = 'train'
            
        local_img_path = os.path.join(raw_data_dir, split_subfolder, 'images', filename)
        local_mask_path = os.path.join(raw_data_dir, split_subfolder, 'masks', filename)
        
        # Parse coordinates ("1576,1276" -> y=1576, x=1276)
        coord_str = str(row['coordinates']).strip('"\' ')
        coords = [int(c.strip()) for c in coord_str.split(',')]
        coord_y, coord_x = coords[0], coords[1]
        
        class_label = float(row['class'])
        
        records.append({
            'image_path': local_img_path,
            'mask_path': local_mask_path,
            'scene_name': scene_name,
            'coord_y': coord_y,
            'coord_x': coord_x,
            'class_label': class_label
        })
        
    return pd.DataFrame(records)
