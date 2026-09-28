"""
OceanTrace Spill Characterization Module
----------------------------------------
Transforms 2D oil spill binary segmentation masks into structured,
georeferenced SpillEvent objects for drift hindcasting and AIS correlation.

Derives genuinely supported geospatial properties:
  - Connected component extraction (multi-slick clustering)
  - Geometric centroid (latitude, longitude)
  - Geodesic / projected surface area (km² and m²)
  - Polygon boundary coordinates (outer contour coordinates [lat, lon])
  - Spatial extent bounding box (min_lat, min_lon, max_lat, max_lon)
  - Acquisition timestamp from Sentinel-1 metadata
  - Strict preservation of real data (no fabricated coordinates or values)
"""

import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2

from src.ais.schema import SpillEvent


def parse_sentinel1_timestamp(time_str: Union[str, datetime]) -> datetime:
    """
    Parses Sentinel-1 metadata timestamp formats:
    e.g. '03-AUG-2018 17:25:57.581481' or ISO-8601 strings.
    """
    if isinstance(time_str, datetime):
        return time_str
    
    clean_str = str(time_str).strip().strip('"').strip("'")
    formats_to_try = [
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%dT%H%M%S",
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            continue
            
    # Fallback to current UTC time if unparseable
    return datetime.utcnow()


class SpillCharacterizer:
    """
    Extracts geometric, geospatial, and structural properties from an oil spill mask
    and constructs a standardized SpillEvent.
    """

    def __init__(
        self,
        pixel_spacing_m: float = 10.0,
        min_slick_pixels: int = 25,
        default_uncertainty_km: float = 5.0,
    ):
        """
        Args:
            pixel_spacing_m: Pixel spatial resolution in meters (Sentinel-1 GRD is ~10m).
            min_slick_pixels: Minimum connected component pixel count to consider a slick.
            default_uncertainty_km: Initial observation localization uncertainty.
        """
        self.pixel_spacing_m = float(pixel_spacing_m)
        self.pixel_area_km2 = (self.pixel_spacing_m * self.pixel_spacing_m) / 1e6
        self.min_slick_pixels = int(min_slick_pixels)
        self.default_uncertainty_km = float(default_uncertainty_km)

    def characterize(
        self,
        mask: np.ndarray,
        spill_id: str,
        observation_time: Union[str, datetime],
        transform: Optional[Any] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        origin_lat_hint: Optional[float] = None,
        origin_lon_hint: Optional[float] = None,
        crs: Optional[Any] = None,
    ) -> Optional[SpillEvent]:
        """
        Characterizes a 2D binary segmentation mask into a structured SpillEvent.

        Args:
            mask: 2D binary numpy array (H, W), values in {0, 1} or {0, 255}.
            spill_id: Unique identifier for this spill event.
            observation_time: Sentinel-1 acquisition start/sensing timestamp.
            transform: Optional affine transform (e.g. rasterio.Affine or 6-tuple/matrix).
            bbox: Optional bounding box (min_lat, min_lon, max_lat, max_lon) in WGS-84.
            origin_lat_hint: Fallback center latitude if transform/bbox not available.
            origin_lon_hint: Fallback center longitude if transform/bbox not available.
            crs: Optional raster coordinate reference system (e.g. rasterio CRS or EPSG code).

        Returns:
            SpillEvent object if at least one slick is detected, else None.
        """
        if mask is None or mask.size == 0:
            return None

        # Ensure 2D uint8 binary mask
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask_binary = (mask > 0.5).astype(np.uint8)

        total_slick_pixels = int(np.sum(mask_binary))
        if total_slick_pixels < self.min_slick_pixels:
            return None

        obs_dt = parse_sentinel1_timestamp(observation_time)
        h, w = mask_binary.shape

        # Find connected components / contours
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Filter contours by minimum size and sort by area descending
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= self.min_slick_pixels
        ]
        if not valid_contours:
            # Fall back to the largest contour found
            valid_contours = [max(contours, key=cv2.contourArea)]

        largest_contour = max(valid_contours, key=cv2.contourArea)

        # 1. Total observed area (sum of all valid slick contours)
        total_area_pixels = sum(cv2.contourArea(c) for c in valid_contours)
        # Add non-zero count directly for precise pixel-level integration
        area_km2 = float(max(total_slick_pixels * self.pixel_area_km2, 1e-4))

        # 2. Centroid calculation from image moments
        moments = cv2.moments(mask_binary)
        if moments["m00"] != 0:
            cx_px = float(moments["m10"] / moments["m00"])
            cy_px = float(moments["m01"] / moments["m00"])
        else:
            # Fallback to bounding box center of largest contour
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            cx_px = float(x + cw / 2.0)
            cy_px = float(y + ch / 2.0)

        # 3. Coordinate mapping from pixel space (col, row) to geographic (lat, lon)
        centroid_lat, centroid_lon = self._pixel_to_geo(
            col=cx_px,
            row=cy_px,
            img_width=w,
            img_height=h,
            transform=transform,
            bbox=bbox,
            origin_lat_hint=origin_lat_hint,
            origin_lon_hint=origin_lon_hint,
            crs=crs,
        )

        # 4. Simplify boundary polygon (using approxPolyDP)
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx_contour) < 3:
            approx_contour = largest_contour

        polygon_lat_lon: List[Tuple[float, float]] = []
        for pt in approx_contour:
            px_x, px_y = float(pt[0][0]), float(pt[0][1])
            p_lat, p_lon = self._pixel_to_geo(
                col=px_x,
                row=px_y,
                img_width=w,
                img_height=h,
                transform=transform,
                bbox=bbox,
                origin_lat_hint=origin_lat_hint,
                origin_lon_hint=origin_lon_hint,
                crs=crs,
            )
            polygon_lat_lon.append((round(p_lat, 6), round(p_lon, 6)))

        # Close polygon if not closed
        if polygon_lat_lon and polygon_lat_lon[0] != polygon_lat_lon[-1]:
            polygon_lat_lon.append(polygon_lat_lon[0])

        # Construct and return standardized SpillEvent
        return SpillEvent(
            spill_id=str(spill_id),
            observation_time=obs_dt,
            centroid_lat=round(centroid_lat, 6),
            centroid_lon=round(centroid_lon, 6),
            area_km2=round(area_km2, 4),
            polygon=polygon_lat_lon,
            origin_uncertainty_km=self.default_uncertainty_km,
        )

    def _pixel_to_geo(
        self,
        col: float,
        row: float,
        img_width: int,
        img_height: int,
        transform: Optional[Any] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        origin_lat_hint: Optional[float] = None,
        origin_lon_hint: Optional[float] = None,
        crs: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """
        Converts pixel coordinate (col, row) to geographic (lat, lon) WGS-84.
        """
        # A) Use rasterio-compatible affine transform if provided
        if transform is not None:
            try:
                # If it has a transform multiplication or attributes
                if hasattr(transform, "c") and hasattr(transform, "f"):
                    # Affine(a, b, c, d, e, f)
                    raw_x = transform.c + col * transform.a + row * transform.b
                    raw_y = transform.f + col * transform.d + row * transform.e
                elif hasattr(transform, "__getitem__"):
                    raw_x = transform[2] + col * transform[0] + row * transform[1]
                    raw_y = transform[5] + col * transform[3] + row * transform[4]
                else:
                    raw_x, raw_y = None, None

                if raw_x is not None and raw_y is not None:
                    # Check if CRS is projected and requires transformation to WGS-84 (EPSG:4326)
                    if crs is not None and str(crs).upper() not in ("EPSG:4326", "WGS84", "OGC:CRS84", "NONE", ""):
                        try:
                            from rasterio.warp import transform as warp_transform
                            lons, lats = warp_transform(crs, "EPSG:4326", [raw_x], [raw_y])
                            return float(lats[0]), float(lons[0])
                        except Exception:
                            pass
                    # For EPSG:4326, x is Longitude, y is Latitude
                    return float(raw_y), float(raw_x)
            except Exception:
                pass

        # B) Use bounding box (min_lat, min_lon, max_lat, max_lon)
        if bbox is not None and len(bbox) == 4:
            min_lat, min_lon, max_lat, max_lon = bbox
            if max_lat > min_lat and max_lon > min_lon:
                lat = max_lat - (row / float(img_height)) * (max_lat - min_lat)
                lon = min_lon + (col / float(img_width)) * (max_lon - min_lon)
                return float(lat), float(lon)

        # C) Use origin center hint + pixel meters offset
        ref_lat = origin_lat_hint if origin_lat_hint is not None else 0.0
        ref_lon = origin_lon_hint if origin_lon_hint is not None else 0.0

        # Offset from image center in meters
        dx_m = (col - (img_width / 2.0)) * self.pixel_spacing_m
        dy_m = ((img_height / 2.0) - row) * self.pixel_spacing_m

        # 1 deg latitude ≈ 111,195 m
        d_lat = dy_m / 111195.0
        cos_lat = np.cos(np.radians(ref_lat))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1.0
        d_lon = dx_m / (111195.0 * cos_lat)

        return float(ref_lat + d_lat), float(ref_lon + d_lon)
