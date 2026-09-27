"""
Geographic Coordinate Transformations and Geodesic Displacement
----------------------------------------------------------------
Computes latitude/longitude displacements from metric vector velocities,
handles hemisphere boundaries and longitudinal meridian wrapping.
"""

import math
from typing import Tuple

# WGS-84 Mean Earth Radius in meters
EARTH_RADIUS_METERS = 6371008.8


def displace_lat_lon(
    lat: float,
    lon: float,
    dx_meters: float,
    dy_meters: float
) -> Tuple[float, float]:
    """
    Displaces a (lat, lon) coordinate by (dx, dy) in meters.
    
    Args:
        lat: Initial latitude in degrees [-90.0, +90.0]
        lon: Initial longitude in degrees [-180.0, +180.0]
        dx_meters: Displacement Eastward (positive) or Westward (negative) in meters
        dy_meters: Displacement Northward (positive) or Southward (negative) in meters
        
    Returns:
        (new_lat: float, new_lon: float) normalized within valid ranges.
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of bounds [-90, +90]: {lat}")

    # Latitude displacement
    d_lat_deg = (dy_meters / EARTH_RADIUS_METERS) * (180.0 / math.pi)
    new_lat = lat + d_lat_deg

    # Clamp to poles
    new_lat = max(-90.0, min(90.0, new_lat))

    # Longitude displacement scaled by cos(lat)
    cos_lat = math.cos(math.radians(new_lat))
    if abs(cos_lat) < 1e-6:
        # At poles, longitude change is degenerate
        new_lon = lon
    else:
        d_lon_deg = (dx_meters / (EARTH_RADIUS_METERS * cos_lat)) * (180.0 / math.pi)
        new_lon = lon + d_lon_deg

    # Normalize longitude to [-180.0, +180.0]
    new_lon = (new_lon + 180.0) % 360.0 - 180.0

    return new_lat, new_lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return (EARTH_RADIUS_METERS / 1000.0) * c
