"""
GeoJSON and Visual Plotting Utilities for Oil Spill Hindcasting
--------------------------------------------------------------
Exports backward trajectories and origin uncertainty circles to GeoJSON (EPSG:4326)
and renders multi-layer spatial maps.
"""

import os
import json
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hindcasting.schema import HindcastResult


def export_hindcast_to_geojson(result: HindcastResult, output_path: str) -> str:
    """
    Exports the observed spill location, backward trajectory, estimated origin,
    and uncertainty radius circle to an RFC 7946 GeoJSON FeatureCollection.
    """
    features = []

    # 1. Observed Location Point
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [result.observed_lon, result.observed_lat]
        },
        "properties": {
            "entity": "observed_spill",
            "spill_id": result.spill_id,
            "observation_time": result.observation_time.isoformat()
        }
    })

    # 2. Estimated Origin Point
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [result.estimated_origin_lon, result.estimated_origin_lat]
        },
        "properties": {
            "entity": "estimated_origin",
            "model_name": result.model_name,
            "origin_time_start": result.origin_time_start.isoformat(),
            "origin_time_end": result.origin_time_end.isoformat(),
            "uncertainty_km": result.origin_uncertainty_km,
            "drift_duration_hours": result.drift_duration_hours
        }
    })

    # 3. Backward Trajectory LineString
    if len(result.trajectory) >= 2:
        coords = [[p.longitude, p.latitude] for p in result.trajectory]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "entity": "backward_drift_trajectory",
                "drift_hours": result.drift_duration_hours,
                "total_points": len(result.trajectory)
            }
        })

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)

    return output_path


def plot_hindcast_trajectory(result: HindcastResult, output_png: str) -> str:
    """
    Renders a clear, explainable plot showing the observed slick location,
    the backwards drift trajectory arrows, and the origin uncertainty zone.
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    # Plot observed location
    ax.scatter(
        [result.observed_lon], [result.observed_lat],
        color="darkblue", s=130, zorder=6, marker="o", label="Observed Slick (SAR Detection)"
    )

    # Plot backward trajectory
    if len(result.trajectory) >= 2:
        lons = [p.longitude for p in result.trajectory]
        lats = [p.latitude for p in result.trajectory]
        ax.plot(
            lons, lats,
            color="royalblue", linestyle="--", linewidth=2.0, zorder=4,
            label=f"Backward Drift Track ({result.drift_duration_hours}h)"
        )
        # Mark intermediate hours
        for p in result.trajectory[1:-1]:
            ax.scatter([p.longitude], [p.latitude], color="cornflowerblue", s=30, zorder=5)

    # Plot estimated origin
    ax.scatter(
        [result.estimated_origin_lon], [result.estimated_origin_lat],
        color="crimson", s=180, zorder=7, marker="*", label=f"Estimated Origin ({result.model_name})"
    )

    # Draw uncertainty circle approximation
    deg_radius = result.origin_uncertainty_km / 111.0
    circle = plt.Circle(
        (result.estimated_origin_lon, result.estimated_origin_lat),
        deg_radius, color="red", fill=True, alpha=0.15, linestyle=":",
        linewidth=1.5, label=f"Origin Uncertainty (±{result.origin_uncertainty_km} km)"
    )
    ax.add_patch(circle)

    ax.set_title(f"OceanTrace Backward Drift Hindcast (Spill: {result.spill_id})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (deg)", fontsize=10)
    ax.set_ylabel("Latitude (deg)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    return output_png
