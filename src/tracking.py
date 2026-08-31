"""
OceanTrace - Prototype Drift & Trajectory Tracking Module

Provides prototype slick displacement, centroid calculation, and drift trajectory
demonstration for time-series satellite decision support.
"""

import numpy as np
import cv2


def simulate_slick_drift(
    binary_mask: np.ndarray,
    wind_speed_knots: float = 12.0,
    wind_direction_deg: float = 45.0,
    current_speed_knots: float = 1.5,
    forecast_hours: int = 6
) -> dict:
    """
    Prototype Drift Vector Calculation (Prototype Simulation).
    
    Formula:
    Net Drift Velocity = (0.03 * Wind Velocity) + (1.0 * Ocean Current Velocity)
    
    Args:
        binary_mask (np.ndarray): Binary prediction mask.
        wind_speed_knots (float): Estimated wind speed in knots.
        wind_direction_deg (float): Wind direction in degrees.
        current_speed_knots (float): Ocean current speed in knots.
        forecast_hours (int): Time horizon for forward drift simulation.
        
    Returns:
        dict: {
            'centroid': (cy, cx),
            'projected_centroid': (p_cy, p_cx),
            'drift_distance_nm': float,
            'drift_heading_deg': float,
            'trajectory_pts': list of tuples
        }
    """
    coords = np.argwhere(binary_mask > 0)
    if len(coords) == 0:
        return {
            'centroid': None,
            'projected_centroid': None,
            'drift_distance_nm': 0.0,
            'drift_heading_deg': 0.0,
            'trajectory_pts': []
        }
        
    cy, cx = np.mean(coords, axis=0)
    
    # Net drift vector calculation (knots)
    wind_rad = np.radians(wind_direction_deg)
    net_u = (0.03 * wind_speed_knots + current_speed_knots) * np.sin(wind_rad)
    net_v = (0.03 * wind_speed_knots + current_speed_knots) * np.cos(wind_rad)
    
    drift_distance_nm = np.sqrt(net_u**2 + net_v**2) * forecast_hours
    drift_heading_deg = (np.degrees(np.arctan2(net_u, net_v)) + 360) % 360
    
    # Calculate trajectory points in image pixel space (scaling 1 nm = 5 pixels for prototype rendering)
    scale_nm_to_pixels = 5.0
    trajectory_pts = []
    
    for h in range(0, forecast_hours + 1):
        step_dist = (drift_distance_nm * (h / max(1, forecast_hours))) * scale_nm_to_pixels
        p_cx = int(cx + step_dist * np.sin(wind_rad))
        p_cy = int(cy - step_dist * np.cos(wind_rad))
        trajectory_pts.append((p_cx, p_cy))
        
    projected_centroid = trajectory_pts[-1]
    
    return {
        'centroid': (int(cx), int(cy)),
        'projected_centroid': projected_centroid,
        'drift_distance_nm': round(float(drift_distance_nm), 2),
        'drift_heading_deg': round(float(drift_heading_deg), 1),
        'trajectory_pts': trajectory_pts
    }


def draw_drift_trajectory(
    image: np.ndarray,
    binary_mask: np.ndarray,
    drift_info: dict
) -> np.ndarray:
    """
    Draws drift trajectory vectors on prediction image canvas (Prototype Simulation).
    """
    vis_img = image.copy()
    if vis_img.dtype != np.uint8:
        vis_img = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)
        
    if vis_img.ndim == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    elif vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
    pts = drift_info.get('trajectory_pts', [])
    if len(pts) >= 2:
        # Draw trajectory line
        for i in range(len(pts) - 1):
            cv2.line(vis_img, pts[i], pts[i+1], (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis_img, pts[i], 3, (0, 255, 255), -1)
            
        # Draw end arrow
        cv2.arrowedLine(vis_img, pts[-2], pts[-1], (0, 0, 255), 3, tipLength=0.3)
        cv2.circle(vis_img, pts[0], 5, (0, 255, 0), -1) # Start centroid green
        
        # Add label
        label = f"Drift: {drift_info['drift_distance_nm']} NM @ {drift_info['drift_heading_deg']} deg"
        cv2.putText(vis_img, label, (pts[-1][0] + 10, pts[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
