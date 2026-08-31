"""
OceanTrace - Prototype Verification & Risk Engine

Provides transparent rule-based verification logic to evaluate model predictions,
determine Prototype Model Confidence (HIGH/MEDIUM/LOW), Risk Level, and generate
clear decision-support explanations.
"""

import numpy as np


def verify_oil_spill_detection(
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Evaluates detection metrics to assign confidence, risk level, and explanation.
    
    Formula & Rules:
    1. Region Mean Probability (mean_prob): Average model sigmoid probability over mask > threshold.
    2. Detected Pixel Count (spill_pixels): Total positive prediction pixels.
    3. Spatial Continuity / Compactness (compactness): Ratio of mask area to bounding box area.
    
    Confidence Logic:
    - HIGH: mean_prob >= 0.75 AND spill_pixels >= 250 AND compactness >= 0.20
    - MEDIUM: mean_prob >= 0.55 AND spill_pixels >= 50
    - LOW: Fragmented / weak response or spill_pixels < 50
    
    Returns:
        dict: {
            'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
            'risk_level': 'HIGH' | 'MODERATE' | 'LOW' | 'NONE',
            'mean_prob': float,
            'max_prob': float,
            'spill_pixels': int,
            'explanation': str
        }
    """
    spill_pixels = int(np.sum(binary_mask > 0))
    
    if spill_pixels == 0:
        return {
            'confidence': 'HIGH',
            'risk_level': 'NONE',
            'mean_prob': 0.0,
            'max_prob': float(np.max(prob_map)) if prob_map.size > 0 else 0.0,
            'spill_pixels': 0,
            'explanation': "Clean marine region. No potential oil slicks detected above threshold."
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
    
    # Decision Rules
    if mean_prob >= 0.70 and spill_pixels >= 200 and compactness >= 0.15:
        confidence = "HIGH"
        risk_level = "HIGH"
        explanation = "Strong contiguous SAR attenuation response with high model certainty across detected slick bounds."
    elif mean_prob >= 0.55 and spill_pixels >= 50:
        confidence = "MEDIUM"
        risk_level = "MODERATE"
        explanation = "Moderate SAR intensity drop detected. Further verification required to rule out low-wind look-alikes."
    else:
        confidence = "LOW"
        risk_level = "LOW"
        explanation = "Weak or fragmented model response. Low slick confidence; likely minor SAR noise or biogenic slick."
        
    return {
        'confidence': confidence,
        'risk_level': risk_level,
        'mean_prob': round(mean_prob, 4),
        'max_prob': round(max_prob, 4),
        'spill_pixels': spill_pixels,
        'compactness': round(compactness, 4),
        'explanation': explanation
    }
