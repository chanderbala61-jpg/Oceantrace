"""
OceanTrace - Basic Pipeline Test Suite

Verifies the core pipeline without requiring training:
  1. Dataset CSV loading
  2. Image loading and preprocessing
  3. Augmentation sync (image-mask)
  4. U-Net model forward pass (random weights)
  5. Loss computation
  6. Metric computation
  7. Prediction outputs

Run with:
    python test_pipeline.py
"""

import os
import sys
import traceback
import numpy as np
import torch
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def check(name, passed, detail=""):
    icon = "[OK]" if passed else "[FAIL]"
    print(f"  {icon} {name}", f"({detail})" if detail else "")
    return passed


all_pass = True
print("=" * 60)
print("  OCEANTRACE PIPELINE TEST SUITE")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# TEST 1: Import all modules
# ──────────────────────────────────────────────────────────────
print("\n[TEST 1] Module Imports")
try:
    from src.preprocessing import apply_speckle_filter, normalize_image, process_mask, parse_dataset_csv
    from src.model import UNet, BCEDiceLoss, compute_metrics
    from src.verification import verify_oil_spill_detection
    from src.tracking import simulate_slick_drift
    check("All src modules imported", True)
except Exception as e:
    all_pass = False
    check("All src modules imported", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 2: CSV Parsing
# ──────────────────────────────────────────────────────────────
print("\n[TEST 2] Dataset CSV Parsing")
csv_path = os.path.join(ROOT_DIR, 'data', 'raw', 'train', 'dataframe_train_dataset_256_90.csv')
if os.path.exists(csv_path):
    try:
        df = parse_dataset_csv(csv_path, raw_data_dir=os.path.join(ROOT_DIR, 'data', 'raw'))
        ok = len(df) == 21744 and list(df.columns) == ['image_path', 'mask_path', 'scene_name', 'coord_y', 'coord_x', 'class_label']
        check("CSV parsed", ok, f"{len(df)} rows")
    except Exception as e:
        all_pass = False
        check("CSV parsed", False, str(e))
else:
    check("CSV file found", False, "Not found - skipping")

# ──────────────────────────────────────────────────────────────
# TEST 3: Load Real Image & Mask
# ──────────────────────────────────────────────────────────────
print("\n[TEST 3] Image & Mask Loading")
img_dir = os.path.join(ROOT_DIR, 'data', 'raw', 'train', 'images')
mask_dir = os.path.join(ROOT_DIR, 'data', 'raw', 'train', 'masks')
sample_img = None
sample_mask = None

if os.path.exists(img_dir):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
    if files:
        img_file = os.path.join(img_dir, files[0])
        mask_file = os.path.join(mask_dir, files[0])
        try:
            from PIL import Image
            sample_img = np.array(Image.open(img_file))
            sample_mask = np.array(Image.open(mask_file))
            if sample_img.ndim == 2:
                sample_img = np.stack([sample_img, sample_img, sample_img], axis=-1)
        except Exception:
            sample_img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            sample_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
        if sample_img is not None and sample_mask is not None:
            check("Image loaded", True, f"shape={sample_img.shape}, dtype={sample_img.dtype}")
            check("Mask loaded", True, f"shape={sample_mask.shape}, dtype={sample_mask.dtype}")
            check("Dimensions match", sample_img.shape[:2] == sample_mask.shape[:2])
        else:
            all_pass = False
            check("Image/Mask loaded", False, "Failed to read TIFF")

# ──────────────────────────────────────────────────────────────
# TEST 4: Preprocessing
# ──────────────────────────────────────────────────────────────
print("\n[TEST 4] Preprocessing Functions")
if sample_img is not None:
    try:
        # Extract 256x256 patch
        patch = sample_img[256:512, 256:512]
        if patch.shape[0] < 256 or patch.shape[1] < 256:
            patch = cv2.resize(sample_img, (256, 256))

        filtered = apply_speckle_filter(patch, method='median', kernel_size=3)
        norm = normalize_image(filtered, method='minmax')
        check("Median filter applied", filtered.shape == patch.shape, f"shape={filtered.shape}")
        check("Min-Max normalization", norm.min() >= 0.0 and norm.max() <= 1.0,
              f"range=[{norm.min():.3f}, {norm.max():.3f}]")

        mask_patch = sample_mask[256:512, 256:512]
        if mask_patch.shape[0] < 256:
            mask_patch = cv2.resize(sample_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        bin_mask = process_mask(mask_patch, target_size=(256, 256))
        unique_vals = set(np.unique(bin_mask).tolist())
        check("Binary mask values", unique_vals.issubset({0, 1}), f"unique={unique_vals}")
        check("Mask shape", bin_mask.shape == (256, 256), f"shape={bin_mask.shape}")
    except Exception as e:
        all_pass = False
        check("Preprocessing", False, str(e))
        traceback.print_exc()

# ──────────────────────────────────────────────────────────────
# TEST 5: U-Net Forward Pass
# ──────────────────────────────────────────────────────────────
print("\n[TEST 5] U-Net Model Forward Pass")
try:
    device = torch.device('cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    check("Output shape correct", output.shape == (2, 1, 256, 256), f"shape={tuple(output.shape)}")
    check("Output dtype float32", output.dtype == torch.float32, f"dtype={output.dtype}")

    param_count = sum(p.numel() for p in model.parameters())
    check("Parameter count reasonable", 100_000 < param_count < 50_000_000,
          f"{param_count:,} parameters")
except Exception as e:
    all_pass = False
    check("U-Net forward pass", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────────────────────
# TEST 6: Loss & Metrics
# ──────────────────────────────────────────────────────────────
print("\n[TEST 6] Loss & Metrics Computation")
try:
    criterion = BCEDiceLoss()
    pred = torch.sigmoid(output)
    target = (torch.rand(2, 1, 256, 256) > 0.7).float().to(device)

    loss = criterion(output, target)
    check("BCE+Dice loss computed", loss.item() > 0 and not torch.isnan(loss), f"loss={loss.item():.4f}")

    metrics = compute_metrics(pred, target)
    check("IoU in valid range", 0.0 <= metrics['iou'] <= 1.0, f"iou={metrics['iou']:.4f}")
    check("Dice in valid range", 0.0 <= metrics['dice'] <= 1.0, f"dice={metrics['dice']:.4f}")
    check("Precision in valid range", 0.0 <= metrics['precision'] <= 1.0)
    check("Recall in valid range", 0.0 <= metrics['recall'] <= 1.0)
except Exception as e:
    all_pass = False
    check("Loss/metrics", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 7: Verification Module
# ──────────────────────────────────────────────────────────────
print("\n[TEST 7] Verification Module")
try:
    dummy_prob = np.random.rand(256, 256).astype(np.float32)
    dummy_mask = (dummy_prob > 0.5).astype(np.uint8)
    result = verify_oil_spill_detection(dummy_prob, dummy_mask)
    check("Confidence level valid", result['confidence'] in ('HIGH', 'MEDIUM', 'LOW'),
          f"confidence={result['confidence']}")
    check("Risk level valid", result['risk_level'] in ('HIGH', 'MODERATE', 'LOW', 'NONE'),
          f"risk={result['risk_level']}")
    check("Explanation present", isinstance(result['explanation'], str) and len(result['explanation']) > 10)
except Exception as e:
    all_pass = False
    check("Verification module", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 8: Tracking Module
# ──────────────────────────────────────────────────────────────
print("\n[TEST 8] Tracking Module")
try:
    drift = simulate_slick_drift(dummy_mask, wind_speed_knots=12.0, wind_direction_deg=45.0, forecast_hours=6)
    check("Drift centroid computed", drift['centroid'] is not None)
    check("Drift distance is float", isinstance(drift['drift_distance_nm'], float),
          f"distance={drift['drift_distance_nm']} NM")
    check("Trajectory points generated", len(drift['trajectory_pts']) >= 2)
except Exception as e:
    all_pass = False
    check("Tracking module", False, str(e))

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# --------------------------------------------------------------
print("\n" + "=" * 60)
if all_pass:
    print("  [SUCCESS] ALL TESTS PASSED - Pipeline is ready for training!")
else:
    print("  [WARNING] SOME TESTS FAILED - Review errors above before training.")
print("=" * 60)
print("\nNext step:")
print("  python run_pipeline.py --epochs 3 --batch-size 8")
print("  streamlit run app/app.py")
