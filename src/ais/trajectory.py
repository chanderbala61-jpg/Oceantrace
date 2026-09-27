"""
Vessel Trajectory Reconstruction and Spatial-Temporal Interpolation
-------------------------------------------------------------------
Groups AIS records by MMSI, sorts them chronologically, and calculates
closest point of approach (CPA) to a target spill origin.
"""

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from src.ais.schema import AISRecord
from src.ais.spatial import haversine_distance_km, closest_distance_to_segment_km


class VesselTrajectory:
    """
    Represents the ordered trajectory of a single vessel (one MMSI) over time.
    """
    def __init__(self, mmsi: int, records: List[AISRecord]):
        self.mmsi = mmsi
        # Chronological sort
        self.records = sorted(records, key=lambda r: r.timestamp)
        self.ship_name = self.records[0].ship_name if self.records else None
        self.ship_type = self.records[0].ship_type if self.records else None

    @property
    def num_points(self) -> int:
        return len(self.records)

    @property
    def start_time(self) -> Optional[datetime]:
        return self.records[0].timestamp if self.records else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.records[-1].timestamp if self.records else None

    @property
    def duration_hours(self) -> float:
        if len(self.records) < 2:
            return 0.0
        return (self.records[-1].timestamp - self.records[0].timestamp).total_seconds() / 3600.0

    def compute_closest_approach(
        self,
        target_lat: float,
        target_lon: float
    ) -> Dict[str, Any]:
        """
        Computes the closest approach distance and time between the vessel trajectory
        and a target point (such as estimated spill origin or observed centroid).
        
        Evaluates both individual broadcast pings and linear path segments between pings.
        """
        if not self.records:
            return {
                "min_distance_km": float("inf"),
                "closest_ping_time": None,
                "closest_lat": None,
                "closest_lon": None,
                "interpolated": False
            }

        # 1. Point-wise minimum
        best_dist = float("inf")
        best_time = self.records[0].timestamp
        best_lat = self.records[0].latitude
        best_lon = self.records[0].longitude
        interpolated = False

        for r in self.records:
            d = haversine_distance_km(r.latitude, r.longitude, target_lat, target_lon)
            if d < best_dist:
                best_dist = d
                best_time = r.timestamp
                best_lat = r.latitude
                best_lon = r.longitude

        # 2. Segment-wise interpolation for consecutive pings separated by < 2 hours
        for i in range(len(self.records) - 1):
            r1 = self.records[i]
            r2 = self.records[i + 1]
            dt_sec = (r2.timestamp - r1.timestamp).total_seconds()
            
            # Interpolate only for reasonably continuous tracks (<= 2 hours gap)
            if 0 < dt_sec <= 7200:
                seg_dist, c_lat, c_lon = closest_distance_to_segment_km(
                    target_lat, target_lon,
                    r1.latitude, r1.longitude,
                    r2.latitude, r2.longitude
                )
                if seg_dist < best_dist:
                    best_dist = seg_dist
                    best_lat = c_lat
                    best_lon = c_lon
                    # Linearly interpolate time
                    total_seg_len = haversine_distance_km(r1.latitude, r1.longitude, r2.latitude, r2.longitude)
                    if total_seg_len > 0:
                        frac = haversine_distance_km(r1.latitude, r1.longitude, c_lat, c_lon) / total_seg_len
                        frac = max(0.0, min(1.0, frac))
                    else:
                        frac = 0.0
                    best_time = r1.timestamp + (r2.timestamp - r1.timestamp) * frac
                    interpolated = True

        return {
            "min_distance_km": float(best_dist),
            "closest_approach_time": best_time,
            "closest_lat": float(best_lat),
            "closest_lon": float(best_lon),
            "interpolated": interpolated
        }


def build_vessel_trajectories(records: List[AISRecord]) -> Dict[int, VesselTrajectory]:
    """
    Groups a flat list of AISRecords by MMSI and constructs VesselTrajectory objects.
    """
    grouped: Dict[int, List[AISRecord]] = {}
    for r in records:
        if r.mmsi not in grouped:
            grouped[r.mmsi] = []
        grouped[r.mmsi].append(r)

    trajectories = {}
    for mmsi, pings in grouped.items():
        trajectories[mmsi] = VesselTrajectory(mmsi, pings)
    return trajectories
