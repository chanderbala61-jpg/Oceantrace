"""
OceanTrace - PyTorch Dataset Module

Implements OilSpillDataset for loading SAR satellite patches and oil spill target masks
with synchronized Albumentations spatial augmentations.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from src.preprocessing import (
    apply_speckle_filter,
    normalize_image,
    process_mask,
    parse_dataset_csv
)


def get_default_transforms(mode: str = 'train'):
    """
    Returns spatial augmentations for SAR imagery.
    Only applies non-destructive geometric flips/rotations to preserve SAR backscatter physics.
    
    Args:
        mode (str): 'train', 'val', 'validation', or 'test'.
        
    Returns:
        A.Compose: Albumentations composition object.
    """
    if mode.lower() == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([], additional_targets={'mask': 'mask'})


class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for Satellite SAR Oil Spill Detection.
    
    Supports patch loading from pre-generated CSV coordinate manifests or direct scene indexing.
    """
    def __init__(
        self,
        mode: str = 'train',
        csv_path: str = None,
        raw_data_dir: str = 'data/raw',
        patch_size: tuple = (256, 256),
        stride: int = 90,
        filter_method: str = 'median',
        filter_kernel_size: int = 3,
        norm_method: str = 'minmax',
        use_augmentation: bool = True,
        transform = None
    ):
        """
        Args:
            mode (str): Dataset split mode ('train', 'val', 'validation', 'test').
            csv_path (str): Optional path to annotation CSV file (e.g. dataframe_train_dataset_256_90.csv).
            raw_data_dir (str): Path to raw data directory containing train/ and test/ subdirectories.
            patch_size (tuple): Target patch dimensions (height, width). Default (256, 256).
            stride (int): Patch extraction stride in pixels. Default 90.
            filter_method (str): SAR speckle filter method ('median', 'lee', or 'none'). Default 'median'.
            filter_kernel_size (int): Window size for noise filter. Default 3.
            norm_method (str): Normalization method ('minmax', 'zscore', or 'none'). Default 'minmax'.
            use_augmentation (bool): Whether to apply spatial data augmentations in training mode.
            transform: Optional custom Albumentations transform object.
        """
        self.mode = mode.lower()
        self.raw_data_dir = raw_data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.filter_method = filter_method
        self.filter_kernel_size = filter_kernel_size
        self.norm_method = norm_method
        self.use_augmentation = use_augmentation and (self.mode == 'train')
        
        # Configure Albumentations transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_default_transforms(mode=self.mode) if self.use_augmentation else A.Compose([])
            
        self._scene_cache = {}
        self.samples = []
        
        # Load samples from CSV if provided, else scan directory
        if csv_path is not None and os.path.exists(csv_path):
            df = parse_dataset_csv(csv_path, raw_data_dir=self.raw_data_dir)
            for _, row in df.iterrows():
                self.samples.append({
                    'image_path': row['image_path'],
                    'mask_path': row['mask_path'],
                    'coord_y': row['coord_y'],
                    'coord_x': row['coord_x'],
                    'class_label': row['class_label']
                })
        else:
            self._scan_directory()
            
    def _scan_directory(self):
        """Scans raw_data_dir for scene image and mask pairs."""
        split_folder = 'test' if self.mode == 'test' else 'train'
        img_dir = os.path.join(self.raw_data_dir, split_folder, 'images')
        mask_dir = os.path.join(self.raw_data_dir, split_folder, 'masks')
        
        if not os.path.exists(img_dir):
            return
            
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
        
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                self.samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'coord_y': None,
                    'coord_x': None,
                    'class_label': None
                })
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        sample_info = self.samples[idx]
        img_path = sample_info['image_path']
        mask_path = sample_info['mask_path']
        
        try:
            # Read full image and mask (cached for fast patch slicing)
            if img_path in self._scene_cache:
                image, mask = self._scene_cache[img_path]
            else:
                try:
                    from PIL import Image
                    with Image.open(img_path) as pil_img:
                        image = np.array(pil_img)
                    with Image.open(mask_path) as pil_mask:
                        mask = np.array(pil_mask)
                except Exception:
                    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
                if image is None or mask is None:
                    raise FileNotFoundError(f"Failed to read image or mask from {img_path}")
                    
                # Convert single channel / multi-channel SAR rasters to 3D array (H, W, C)
                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)
                elif image.shape[-1] == 4:
                    image = image[:, :, :3]
                elif image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)
                    
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 0).astype(np.uint8)
                
                # Cache full scene arrays
                if len(self._scene_cache) < 25: # Keep up to 25 scenes in memory
                    self._scene_cache[img_path] = (image, mask)
                
            pH, pW = self.patch_size
            coord_y = sample_info['coord_y']
            coord_x = sample_info['coord_x']
            
            # Crop patch if coordinates specified
            if coord_y is not None and coord_x is not None:
                H, W = image.shape[:2]
                y1 = min(max(0, coord_y), max(0, H - pH))
                x1 = min(max(0, coord_x), max(0, W - pW))
                
                img_patch = image[y1:y1+pH, x1:x1+pW]
                mask_patch = mask[y1:y1+pH, x1:x1+pW]
            else:
                # Resize to target patch size if scene dataset mode
                img_patch = cv2.resize(image, (pW, pH), interpolation=cv2.INTER_LINEAR)
                mask_patch = cv2.resize(mask, (pW, pH), interpolation=cv2.INTER_NEAREST)
                
            # Process mask (binary integer 0 or 1)
            mask_patch = (mask_patch > 0).astype(np.uint8)
            
            # Apply Speckle Filter to Image (Preserve edges)
            img_filtered = apply_speckle_filter(
                img_patch,
                method=self.filter_method,
                kernel_size=self.filter_kernel_size
            )
            
            # Apply Per-Image Min-Max Normalization to [0, 1]
            img_norm = normalize_image(img_filtered, method=self.norm_method)
            
            # Apply Synchronized Albumentations Transforms
            if self.use_augmentation and self.transform is not None:
                augmented = self.transform(image=img_norm, mask=mask_patch)
                img_norm = augmented['image']
                mask_patch = augmented['mask']
                
            # Ensure channel dimensions for PyTorch
            if img_norm.ndim == 2:
                img_norm = np.expand_dims(img_norm, axis=-1)
                
            # Convert to PyTorch Tensors [C, H, W]
            image_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0) # [1, H, W]
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'image_path': img_path,
                'class_label': torch.tensor(sample_info['class_label'] if sample_info['class_label'] is not None else 0.0, dtype=torch.float32)
            }
            
        except Exception as e:
            # Graceful error handling: Return fallback blank tensor with warning
            print(f"[Warning] Error loading index {idx} ({img_path}): {e}")
            fallback_img = torch.zeros((1, self.patch_size[0], self.patch_size[1]), dtype=torch.float32)
            fallback_mask = torch.zeros((1, self.patch_size[0], self.patch_size[1]), dtype=torch.float32)
            return {
                'image': fallback_img,
                'mask': fallback_mask,
                'image_path': img_path,
                'class_label': torch.tensor(0.0, dtype=torch.float32)
            }
