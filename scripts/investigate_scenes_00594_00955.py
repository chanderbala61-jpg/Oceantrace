"""
Investigate Scene 00594 and Scene 00955
- Scene 00594: Limitation analysis (weak contrast, zero GT overlap, AIS coverage gap)
- Scene 00955: Validated workflow (real slick, RK4 hindcast origin, AIS correlation)
"""
import os
import sys
import numpy as np
import rasterio
import tifffile
import torch
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.inference import load_frozen_segformer, predict_scene_tiled, DEFAULT_CHECKPOINT_PATH
from src.characterization import SpillCharacterizer
from src.look_alike import LookAlikeAnalyzer
from src.pipeline import OceanTracePipeline
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting.schema import EnvironmentalVector

def investigate_scene_00594():
    print("=" * 60)
    print("INVESTIGATION: SCENE 00594")
    print("=" * 60)
    img_path = os.path.join(REPO_ROOT, "extracted_dataset", "images", "00594.tif")
    mask_path = os.path.join(REPO_ROOT, "extracted_dataset", "masks", "00594.tif")
    
    with rasterio.open(img_path) as src:
        sar = src.read([1, 2]).transpose(1, 2, 0).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        tags = src.tags()
    
    mask = tifffile.imread(mask_path).astype(np.uint8) if os.path.exists(mask_path) else None
    
    print(f"Image shape: {sar.shape}")
    print(f"Bounds: {bounds}")
    print(f"Transform: {transform}")
    print(f"CRS: {crs}")
    print(f"Tags: {tags}")
    print(f"Ground truth active pixels: {np.sum(mask > 0) if mask is not None else 'None'}")
    
    model, meta = load_frozen_segformer(DEFAULT_CHECKPOINT_PATH, device="cpu", verify_hash=True)
    pred_res = predict_scene_tiled(model, sar, patch_size=256, stride=256)
    pred_mask = pred_res["binary_mask"]
    active_pred = int(np.sum(pred_mask > 0))
    print(f"Predicted active pixels: {active_pred}")
    
    # Characterization
    char = SpillCharacterizer()
    spill = char.characterize(
        mask=pred_mask,
        spill_id="spill_00594",
        observation_time=datetime(2017, 4, 28, 0, 1, 30),
        transform=transform,
        origin_lat_hint=bounds.top if bounds else 24.5,
        origin_lon_hint=bounds.left if bounds else -85.5,
    )
    if spill:
        print(f"Spill Centroid: {spill.centroid_lat:.4f}°N, {spill.centroid_lon:.4f}°E")
        print(f"Spill Area: {spill.area_km2:.4f} km²")
    
    # Look alike
    la = LookAlikeAnalyzer()
    look = la.assess(
        spill_id="spill_00594",
        mask=pred_mask,
        sar_image=sar,
        wind_speed_ms=3.4,
    )
    print(f"Damping Contrast: {look.contrast_db} dB (threshold: {la.min_contrast_db} dB)")
    print(f"Edge Gradient: {look.edge_gradient}")
    print(f"Ambiguity Score: {look.ambiguity_score} [{look.ambiguity_level}]")
    print(f"Classification: {look.classification}")
    print(f"Flags: {look.flags}")
    print(f"Warnings: {look.warnings}")


def investigate_scene_00955():
    print("\n" + "=" * 60)
    print("INVESTIGATION: SCENE 00955")
    print("=" * 60)
    img_path = os.path.join(REPO_ROOT, "extracted_dataset", "images", "00955.tif")
    mask_path = os.path.join(REPO_ROOT, "extracted_dataset", "masks", "00955.tif")
    ais_path = os.path.join(REPO_ROOT, "outputs", "investigations", "00955_ais_traffic.csv")
    
    with rasterio.open(img_path) as src:
        sar = src.read([1, 2]).transpose(1, 2, 0).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    
    mask = tifffile.imread(mask_path).astype(np.uint8) if os.path.exists(mask_path) else None
    print(f"Image shape: {sar.shape}")
    print(f"Bounds: {bounds}")
    print(f"CRS: {crs}")
    print(f"Ground truth active pixels: {np.sum(mask > 0) if mask is not None else 'None'}")
    
    model, meta = load_frozen_segformer(DEFAULT_CHECKPOINT_PATH, device="cpu", verify_hash=True)
    pipe = OceanTracePipeline(model=model)
    
    env_vec = EnvironmentalVector(u=5.5, v=2.8, source="Reanalysis_Wind_Vector")
    env_prov = ConstantEnvironmentalProvider(wind=env_vec)
    
    rec = pipe.execute(
        scene_id="00955",
        sar_image=sar,
        mask=mask,
        observation_time=datetime(2019, 8, 7, 0, 25, 51),
        origin_lat_hint=27.8500,
        origin_lon_hint=-90.5000,
        transform=transform,
        env_provider=env_prov,
        ais_csv_path=ais_path,
        wind_speed_ms=float(np.hypot(5.5, 2.8)),
        drift_hours=6.0,
        model=model,
        run_inference=True,
    )
    
    print(f"Detection Method: {rec.detection_method}")
    print(f"Has Detection: {rec.has_detection}")
    if rec.spill:
        print(f"Observed Area: {rec.spill.area_km2:.4f} km²")
        print(f"Observed Centroid: {rec.spill.centroid_lat:.4f}°N, {rec.spill.centroid_lon:.4f}°E")
    if rec.hindcast_result:
        h = rec.hindcast_result
        print(f"Estimated Probable Origin: {h.estimated_origin_lat:.4f}°N, {h.estimated_origin_lon:.4f}°E")
        print(f"Uncertainty: ±{h.origin_uncertainty_km:.2f} km")
        print(f"Drift Model: {rec.drift_model_name}")
    print(f"AIS Available: {rec.ais_available}")
    print(f"Total Vessels Checked: {rec.total_vessels_checked}")
    print(f"Candidate Count: {len(rec.candidate_vessels)}")
    for cand in rec.candidate_vessels:
        m = cand.measurements
        print(f"  Rank #{cand.rank}: MMSI {cand.mmsi} ({m.get('ship_name', 'N/A')}) - Score: {cand.overall_score:.3f} [{cand.classification}] - CPA: {m.get('min_distance_km', 0):.2f} km")

if __name__ == "__main__":
    investigate_scene_00594()
    investigate_scene_00955()
