"""
Spatial and Trajectory Visualization Utilities for AIS Correlation
------------------------------------------------------------------
Generates:
  - GeoJSON feature collections for GIS integration
  - Visual summary plots showing spill polygon, estimated origin buffer, and AIS vessel tracks
"""

import os
import json
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ais.schema import SpillEvent, CandidateVesselResult
from src.ais.trajectory import VesselTrajectory


def export_correlation_to_geojson(
    spill: SpillEvent,
    trajectories: Dict[int, VesselTrajectory],
    ranked_candidates: List[CandidateVesselResult],
    output_path: str
) -> str:
    """
    Exports the spill detection, origin uncertainty circle, and candidate vessel tracks
    to an RFC 7946 compliant GeoJSON feature collection.
    """
    features = []

    # 1. Observed Centroid Point
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [spill.centroid_lon, spill.centroid_lat]
        },
        "properties": {
            "entity": "observed_spill_centroid",
            "spill_id": spill.spill_id,
            "observation_time": spill.observation_time.isoformat(),
            "area_km2": spill.area_km2
        }
    })

    # 2. Estimated Origin Point (if present)
    if spill.estimated_origin_lat is not None and spill.estimated_origin_lon is not None:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [spill.estimated_origin_lon, spill.estimated_origin_lat]
            },
            "properties": {
                "entity": "estimated_spill_origin",
                "uncertainty_km": spill.origin_uncertainty_km,
                "origin_time_start": spill.origin_time_start.isoformat() if spill.origin_time_start else None,
                "origin_time_end": spill.origin_time_end.isoformat() if spill.origin_time_end else None
            }
        })

    # 3. Candidate Vessel Tracks (LineStrings)
    candidate_map = {c.mmsi: c for c in ranked_candidates}
    for mmsi, traj in trajectories.items():
        if len(traj.records) < 2:
            continue
        coords = [[r.longitude, r.latitude] for r in traj.records]
        cand = candidate_map.get(mmsi)
        
        props = {
            "entity": "vessel_track",
            "mmsi": mmsi,
            "ship_name": traj.ship_name or "Unknown",
            "num_pings": traj.num_points
        }
        if cand:
            props.update({
                "rank": cand.rank,
                "overall_score": cand.overall_score,
                "classification": cand.classification,
                "min_distance_km": cand.measurements.get("min_distance_km")
            })

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": props
        })

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)

    return output_path


def plot_ais_correlation_map(
    spill: SpillEvent,
    trajectories: Dict[int, VesselTrajectory],
    ranked_candidates: List[CandidateVesselResult],
    output_png: str
) -> str:
    """
    Renders an explainable multi-track correlation plot showing spill origin and vessel paths.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot estimated origin
    origin_lat = spill.effective_origin_lat
    origin_lon = spill.effective_origin_lon
    ax.scatter([origin_lon], [origin_lat], color="red", s=120, zorder=5, marker="*", label="Estimated Spill Origin")

    # Draw uncertainty circle approximation
    deg_radius = spill.origin_uncertainty_km / 111.0
    circle = plt.Circle((origin_lon, origin_lat), deg_radius, color="red", fill=False, linestyle="--", linewidth=1.5, label=f"Origin Uncertainty ({spill.origin_uncertainty_km} km)")
    ax.add_patch(circle)

    # Plot candidate tracks
    cmap = plt.cm.get_cmap("tab10")
    for idx, cand in enumerate(ranked_candidates):
        traj = trajectories.get(cand.mmsi)
        if not traj or len(traj.records) < 1:
            continue
        lons = [r.longitude for r in traj.records]
        lats = [r.latitude for r in traj.records]
        color = cmap(idx % 10)
        label = f"Rank {cand.rank}: MMSI {cand.mmsi} (Score: {cand.overall_score:.2f})"
        ax.plot(lons, lats, marker="o", markersize=4, linestyle="-", linewidth=1.8, color=color, label=label)

    ax.set_title(f"OceanTrace AIS Vessel Trajectory Correlation (Spill: {spill.spill_id})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (deg)", fontsize=10)
    ax.set_ylabel("Latitude (deg)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    return output_png
