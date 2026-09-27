"""
Evidence Feature Extraction for Candidate Vessels
-------------------------------------------------
Extracts measurable evidence indicators from vessel trajectories relative to a spill event.
"""

from datetime import datetime
from typing import Dict, Any, List
import math
from src.ais.schema import SpillEvent
from src.ais.trajectory import VesselTrajectory


def extract_candidate_features(
    traj: VesselTrajectory,
    spill: SpillEvent
) -> Dict[str, Any]:
    """
    Computes evidence features for a vessel relative to the spill.
    
    Features computed:
      - min_distance_km: Shortest distance to effective origin
      - closest_approach_time: Datetime of closest approach
      - time_difference_hours: Absolute difference from origin time window center
      - within_uncertainty_zone: Boolean indicating if distance <= origin_uncertainty_km
      - mean_sog: Average speed over ground in knots
      - max_sog: Peak speed over ground in knots
      - sog_variance: Variance of SOG (detects loitering / speed changes)
      - max_reporting_gap_minutes: Longest blackout between consecutive pings
      - ping_count: Number of points recorded
      - trajectory_duration_hours: Total duration of observation
    """
    target_lat = spill.effective_origin_lat
    target_lon = spill.effective_origin_lon
    
    # 1. Spatial closest approach
    approach = traj.compute_closest_approach(target_lat, target_lon)
    min_dist_km = approach["min_distance_km"]
    closest_time = approach["closest_approach_time"]

    # 2. Temporal difference
    if spill.origin_time_start is not None and spill.origin_time_end is not None:
        center_sec = (spill.origin_time_start.timestamp() + spill.origin_time_end.timestamp()) / 2.0
        window_center = datetime.fromtimestamp(center_sec)
    else:
        window_center = spill.observation_time

    if closest_time is not None:
        time_diff_hours = abs((closest_time - window_center).total_seconds()) / 3600.0
    else:
        time_diff_hours = float("inf")

    # 3. Kinematic statistics
    speeds = [r.sog for r in traj.records if r.sog is not None]
    if speeds:
        mean_sog = sum(speeds) / len(speeds)
        max_sog = max(speeds)
        variance_sog = sum((s - mean_sog) ** 2 for s in speeds) / len(speeds)
    else:
        mean_sog, max_sog, variance_sog = 0.0, 0.0, 0.0

    # 4. AIS reporting gaps
    max_gap_min = 0.0
    for i in range(len(traj.records) - 1):
        dt = (traj.records[i + 1].timestamp - traj.records[i].timestamp).total_seconds() / 60.0
        if dt > max_gap_min:
            max_gap_min = dt

    within_uncertainty = min_dist_km <= spill.origin_uncertainty_km

    return {
        "mmsi": traj.mmsi,
        "ship_name": traj.ship_name,
        "ship_type": traj.ship_type,
        "min_distance_km": round(min_dist_km, 3),
        "closest_approach_time": closest_time,
        "closest_lat": approach.get("closest_lat"),
        "closest_lon": approach.get("closest_lon"),
        "interpolated_approach": approach.get("interpolated", False),
        "time_difference_hours": round(time_diff_hours, 2),
        "within_uncertainty_zone": within_uncertainty,
        "mean_sog": round(mean_sog, 2),
        "max_sog": round(max_sog, 2),
        "sog_variance": round(variance_sog, 2),
        "max_reporting_gap_minutes": round(max_gap_min, 1),
        "ping_count": traj.num_points,
        "trajectory_duration_hours": round(traj.duration_hours, 2)
    }
