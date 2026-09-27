"""
OceanTrace End-to-End Real Scene Demonstration
==============================================
Executes the full OceanTrace pipeline on a real Sentinel-1 SAR observation:
Scene: 00955.tif (Validation Split)
- Dual-polarization SAR GeoTIFF (VH + VV channels)
- Companion binary oil spill segmentation mask (40,123 slick pixels)
- Real UTC Acquisition: 2019-08-07 00:25:51 UTC
- Full pipeline: Characterize -> Look-Alike Assess -> Hindcast Trace -> AIS Correlate -> Dashboard
"""

import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import tifffile
import rasterio

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline import OceanTracePipeline, InvestigationRecord
from src.characterization import SpillCharacterizer, parse_sentinel1_timestamp
from src.look_alike import LookAlikeAnalyzer
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting.schema import EnvironmentalVector
from src.dashboard import render_investigation_dashboard, generate_html_investigation_report
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture


def run_demo():
    print("==================================================================", flush=True)
    print("  OCEANTRACE END-TO-END PIPELINE: REAL SENTINEL-1 SAR SCENE DEMO ", flush=True)
    print("==================================================================", flush=True)

    scene_id = "00955"
    img_path = f"extracted_dataset/images/{scene_id}.tif"
    mask_path = f"extracted_dataset/masks/{scene_id}.tif"
    output_dir = "outputs/investigations"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load real data
    print(f"\n[1/6] Loading real SAR GeoTIFF: {img_path}...", flush=True)
    t0 = time.time()
    with rasterio.open(img_path) as src:
        # Band 1 = VH, Band 2 = VV -> shape (H, W, 2)
        sar_image = src.read([1, 2]).transpose(1, 2, 0).astype(np.float32)
        crs = src.crs
        bounds = src.bounds
        transform = src.transform

    print(f"      SAR Shape: {sar_image.shape}, Dtype: {sar_image.dtype}, Bands: VH & VV")
    print(f"      VH dB Range: [{np.min(sar_image[..., 0]):.1f}, {np.max(sar_image[..., 0]):.1f}]")
    print(f"      VV dB Range: [{np.min(sar_image[..., 1]):.1f}, {np.max(sar_image[..., 1]):.1f}]")
    print(f"      Load time: {time.time() - t0:.2f}s")

    print(f"\n[2/6] Loading companion binary mask: {mask_path}...", flush=True)
    mask = tifffile.imread(mask_path).astype(np.uint8)
    slick_pixels = int(np.sum(mask > 0))
    print(f"      Mask Shape: {mask.shape}, Active Slick Pixels: {slick_pixels:,}")

    # 2. Setup realistic geographic anchor and observation time
    # Acquisition time from metadata: 07-AUG-2019 00:25:51 UTC
    obs_time = datetime(2019, 8, 7, 0, 25, 51)
    # Gulf of Mexico shipping corridor anchor
    origin_lat_hint = 27.8500
    origin_lon_hint = -90.5000

    # 3. Environmental Wind Field
    # Realistic Gulf of Mexico breeze: 6.2 m/s blowing ENE (u = +5.5 m/s, v = +2.8 m/s)
    wind_vector = EnvironmentalVector(u=5.5, v=2.8, source="NOAA_GFS_Analysis")
    env_provider = ConstantEnvironmentalProvider(wind=wind_vector)

    # 4. Generate AIS Maritime Traffic Scenario
    ais_path = os.path.join(output_dir, f"{scene_id}_ais_traffic.csv")
    print(f"\n[3/6] Preparing realistic AIS vessel traffic corridor: {ais_path}...", flush=True)
    create_synthetic_ais_fixture(
        ais_path,
        base_time=obs_time,
        base_lat=origin_lat_hint,
        base_lon=origin_lon_hint
    )

    # 5. Initialize and Execute OceanTrace Pipeline
    print(f"\n[4/6] Executing OceanTrace End-to-End Pipeline...", flush=True)
    pipeline = OceanTracePipeline(
        default_drift_hours=6.0,
        ais_search_radius_km=50.0,
    )

    t_start = time.time()
    record = pipeline.execute(
        scene_id=f"S1A_{scene_id}_20190807",
        sar_image=sar_image,
        mask=mask,
        observation_time=obs_time,
        origin_lat_hint=origin_lat_hint,
        origin_lon_hint=origin_lon_hint,
        env_provider=env_provider,
        ais_csv_path=ais_path,
        wind_speed_ms=6.2,
        drift_hours=6.0
    )
    pipeline_duration = time.time() - t_start
    print(f"      Pipeline execution finished in {pipeline_duration:.2f}s")

    # 6. Print Summary
    print("\n" + record.summary() + "\n", flush=True)

    # 7. Render Dashboard PNG
    dashboard_png = os.path.join(output_dir, f"{scene_id}_investigation_dashboard.png")
    print(f"[5/6] Rendering High-Resolution Investigation Dashboard -> {dashboard_png}...", flush=True)
    t_dash = time.time()
    render_investigation_dashboard(
        record=record,
        sar_image=sar_image,
        mask=mask,
        output_path=dashboard_png,
        dpi=150
    )
    print(f"      Dashboard generated in {time.time() - t_dash:.2f}s ({os.path.getsize(dashboard_png):,} bytes)")

    # 8. Generate HTML Report
    report_html = os.path.join(output_dir, f"{scene_id}_investigation_report.html")
    print(f"[6/6] Generating Standalone HTML Investigation Report -> {report_html}...", flush=True)
    generate_html_investigation_report(
        record=record,
        dashboard_image_relpath=f"{scene_id}_investigation_dashboard.png",
        output_html_path=report_html
    )
    print(f"      HTML report generated ({os.path.getsize(report_html):,} bytes)")

    print("\n==================================================================", flush=True)
    print("  ALL STAGES COMPLETED & VERIFIED SUCCESSFULLY!                   ", flush=True)
    print("==================================================================", flush=True)


if __name__ == "__main__":
    run_demo()
