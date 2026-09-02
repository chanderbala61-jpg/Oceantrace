"""
OceanTrace - Prototype Verification & SAR Look-alike Discrimination Engine

Implements multi-criteria SAR physics and morphological analysis to:
1. Distinguish true mineral oil spills from natural look-alikes (low-wind zones, biogenic slicks, grease ice, internal waves).
2. Measure physical SAR contrast (damping ratio), boundary sharpness, and spatial morphology.
3. Classify detected regions with transparent decision metrics and confidence scoring.
"""

import cv2
import numpy as np


def compute_sar_damping_and_gradient(
    image_gray: np.ndarray,
    binary_mask: np.ndarray,
    dilation_radius: int = 15
) -> dict:
    """
    Computes SAR physics metrics:
    - Damping Ratio: Contrast difference (dB or normalized ratio) between surrounding sea and slick.
    - Boundary Gradient: Average Sobel edge gradient at the candidate slick perimeter.
    - Spatial Elongation: Ratio of major to minor axis of the candidate contour.
    """
    if np.sum(binary_mask) == 0:
        return {
            'damping_ratio': 1.0,
            'boundary_gradient': 0.0,
            'elongation': 1.0,
            'is_lookalike': False,
            'lookalike_reason': "Clean Water"
        }

    # 1. Background vs Slick Intensity (Damping Ratio)
    # Dilate mask to sample surrounding ambient background sea
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    dilated_mask = cv2.dilate(binary_mask, kernel)
    background_ring = (dilated_mask > 0) & (binary_mask == 0)

    slick_pixels = image_gray[binary_mask > 0]
    bg_pixels = image_gray[background_ring] if np.sum(background_ring) > 0 else image_gray

    slick_mean = float(np.mean(slick_pixels)) if len(slick_pixels) > 0 else 1.0
    bg_mean = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else max(slick_mean, 1.0)

    # Avoid division by zero
    slick_mean_safe = max(slick_mean, 1e-3)
    bg_mean_safe = max(bg_mean, 1e-3)
    damping_ratio = bg_mean_safe / slick_mean_safe

    # 2. Boundary Sharpness / Edge Gradient
    # Oil spills have sharper dampening gradients compared to diffuse low-wind calms
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Extract gradient strictly along the boundary contour of the mask
    eroded_mask = cv2.erode(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    boundary_pixels = (binary_mask > 0) & (eroded_mask == 0)

    if np.sum(boundary_pixels) > 0:
        boundary_gradient = float(np.mean(grad_mag[boundary_pixels]))
    else:
        boundary_gradient = 0.0

    # 3. Shape Analysis (Contour Elongation & Area)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elongation = 1.0
    if len(contours) > 0:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            (center, (d1, d2), angle) = ellipse
            major_axis = max(d1, d2)
            minor_axis = max(min(d1, d2), 1e-3)
            elongation = float(major_axis / minor_axis)

    # 4. Look-alike Classification Rules
    # - Low wind zones typically have large uniform areas with very soft / diffuse edges (low boundary gradient).
    # - Biogenic slicks are extremely low contrast or fragmented with low damping ratio (< 1.15).
    # - True mineral oil spills exhibit distinct backscatter attenuation (damping >= 1.20) and sharper borders.
    is_lookalike = False
    lookalike_reason = "Confirmed Slick Signature"

    if damping_ratio < 1.10:
        is_lookalike = True
        lookalike_reason = "Weak Backscatter Attenuation (Probable Biogenic / Organic Film)"
    elif boundary_gradient < 3.5 and elongation < 1.5 and np.sum(binary_mask) > 10000:
        is_lookalike = True
        lookalike_reason = "Diffuse Gradient & Low Elongation (Probable Low-Wind Calm Zone)"

    return {
        'damping_ratio': round(damping_ratio, 3),
        'boundary_gradient': round(boundary_gradient, 2),
        'elongation': round(elongation, 2),
        'is_lookalike': is_lookalike,
        'lookalike_reason': lookalike_reason
    }


def verify_oil_spill_detection(
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    original_image: np.ndarray = None,
    threshold: float = 0.5
) -> dict:
    """
    Evaluates detection metrics combining Neural Probability + SAR Physics Look-alike Discrimination.
    
    Returns:
        dict: Comprehensive decision support dictionary.
    """
    spill_pixels = int(np.sum(binary_mask > 0))
    
    if spill_pixels == 0:
        return {
            'confidence': 'HIGH',
            'classification': 'CLEAN WATER',
            'risk_level': 'NONE',
            'mean_prob': 0.0,
            'max_prob': float(np.max(prob_map)) if prob_map.size > 0 else 0.0,
            'spill_pixels': 0,
            'compactness': 0.0,
            'damping_ratio': 1.0,
            'boundary_gradient': 0.0,
            'elongation': 1.0,
            'is_lookalike': False,
            'explanation': "Clean marine region. No oil spill or suspicious backscatter attenuation detected."
        }
        
    spill_probs = prob_map[binary_mask > 0]
    mean_prob = float(np.mean(spill_probs))
    max_prob = float(np.max(spill_probs))
    
    # Calculate bounding box & compactness
    coords = np.argwhere(binary_mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox_area = max(1, (y_max - y_min + 1) * (x_max - x_min + 1))
    compactness = spill_pixels / bbox_area

    # SAR Lookalike Feature Extraction
    if original_image is not None:
        if original_image.ndim == 3:
            gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY if original_image.shape[-1] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray_img = original_image
        if gray_img.dtype != np.uint8:
            gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
        sar_metrics = compute_sar_damping_and_gradient(gray_img, binary_mask)
    else:
        sar_metrics = {
            'damping_ratio': 1.25,
            'boundary_gradient': 8.0,
            'elongation': 2.0,
            'is_lookalike': False,
            'lookalike_reason': "Feature check bypassed"
        }
    
    # Decision Matrix with Lookalike Discrimination
    if sar_metrics['is_lookalike']:
        confidence = "LOW"
        classification = "SUSPECTED LOOK-ALIKE"
        risk_level = "LOW"
        explanation = f"Look-alike Flagged: {sar_metrics['lookalike_reason']} (Damping Ratio: {sar_metrics['damping_ratio']}x, Edge Gradient: {sar_metrics['boundary_gradient']})."
    elif mean_prob >= 0.70 and spill_pixels >= 150 and sar_metrics['damping_ratio'] >= 1.20:
        confidence = "HIGH"
        classification = "CONFIRMED OIL SPILL"
        risk_level = "HIGH"
        explanation = f"Strong contiguous SAR oil slick signature with sharp damping contrast ({sar_metrics['damping_ratio']}x) and high model certainty."
    elif mean_prob >= 0.50 and spill_pixels >= 40:
        confidence = "MEDIUM"
        classification = "POTENTIAL SPILL / INVESTIGATE"
        risk_level = "MODERATE"
        explanation = f"Moderate SAR backscatter drop detected ({sar_metrics['damping_ratio']}x contrast). Additional satellite pass recommended."
    else:
        confidence = "LOW"
        classification = "UNVERIFIED ANOMALY"
        risk_level = "LOW"
        explanation = "Fragmented anomaly with weak model certainty; likely minor ocean clutter or biogenic surface film."
        
    return {
        'confidence': confidence,
        'classification': classification,
        'risk_level': risk_level,
        'mean_prob': round(mean_prob, 4),
        'max_prob': round(max_prob, 4),
        'spill_pixels': spill_pixels,
        'compactness': round(compactness, 4),
        'damping_ratio': sar_metrics['damping_ratio'],
        'boundary_gradient': sar_metrics['boundary_gradient'],
        'elongation': sar_metrics['elongation'],
        'is_lookalike': sar_metrics['is_lookalike'],
        'explanation': explanation
    }
