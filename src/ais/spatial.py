"""
Spatial Geometry and Geodesic Distance Calculations for AIS & Spills
--------------------------------------------------------------------
Provides:
  - haversine_distance_km: Exact great-circle geodesic distance
  - closest_point_on_segment_km: Perpendicular / segment distance to origin
  - calculate_vessel_distances: Distance statistics for a sequence of points
"""

import math
from typing import Tuple, List, Dict, Any


# Earth radius in kilometers (WGS-84 mean radius)
EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two (lat, lon) coordinates in kilometers
    using the numerically stable Haversine formula.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    
    # Clip for floating point safety
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def closest_distance_to_segment_km(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float
) -> Tuple[float, float, float]:
    """
    Approximates the shortest distance from target point P to great-circle segment AB,
    and returns (min_distance_km, closest_lat, closest_lon).
    
    Uses equirectangular local tangent projection for high accuracy over maritime search scales (< 200 km).
    """
    # Reference latitude for projection
    ref_lat_rad = math.radians(p_lat)
    cos_lat = math.cos(ref_lat_rad)

    # Project to planar km coordinates centered at P = (0, 0)
    ax = (a_lon - p_lon) * (math.pi / 180.0) * EARTH_RADIUS_KM * cos_lat
    ay = (a_lat - p_lat) * (math.pi / 180.0) * EARTH_RADIUS_KM
    bx = (b_lon - p_lon) * (math.pi / 180.0) * EARTH_RADIUS_KM * cos_lat
    by = (b_lat - p_lat) * (math.pi / 180.0) * EARTH_RADIUS_KM

    # Vector AB
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        dist = math.sqrt(ax * ax + ay * ay)
        return dist, a_lat, a_lon

    # Parameter t of projection of (0,0) onto segment AB:
    # t = - (A . AB) / |AB|^2 = - (ax*dx + ay*dy) / seg_len_sq
    t = -(ax * dx + ay * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))

    # Closest point coordinates on segment in planar km
    cx = ax + t_clamped * dx
    cy = ay + t_clamped * dy
    dist_km = math.sqrt(cx * cx + cy * cy)

    # Inverse projection to lat/lon
    c_lat = p_lat + (cy / EARTH_RADIUS_KM) * (180.0 / math.pi)
    c_lon = p_lon + (cx / (EARTH_RADIUS_KM * cos_lat)) * (180.0 / math.pi)

    return dist_km, c_lat, c_lon
