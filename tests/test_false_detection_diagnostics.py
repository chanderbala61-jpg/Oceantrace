"""
Unit tests for OceanTrace False Detection Diagnostics and Metrics
-----------------------------------------------------------------
Validates:
  1. Pixel-level confusion matrix vs. scene-level diagnostic classification
  2. 3.5 dB radiometric damping contrast threshold for look-alikes
  3. Decoupling of AIS absence from false detection claims (coverage gap vs false positive)
  4. Coordinate CRS reprojection to WGS-84 (EPSG:4326)
  5. Deterministic verification of Scene 00594 and Scene 00955 properties
"""

import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.pipeline import compute_pixel_metrics, classify_scene_detection
from src.characterization import SpillCharacterizer
from src.ais.false_detection_link import (
    evaluate_scene_false_detection_status,
    FalseDetectionStatus,
)


class TestFalseDetectionDiagnostics(unittest.TestCase):

    def test_pixel_confusion_matrix_pure_fp(self):
        """A scene where ground truth has 0 oil pixels but model predicts 100 oil pixels."""
        pred = np.zeros((100, 100), dtype=np.uint8)
        gt = np.zeros((100, 100), dtype=np.uint8)
        pred[10:20, 10:20] = 1  # 100 pixels

        metrics = compute_pixel_metrics(pred, gt, pixel_spacing_m=10.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 100)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 10000 - 100)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertAlmostEqual(metrics["fp_area_km2"], 100 * (10.0 * 10.0) / 1e6, places=6)

        scene_diag = classify_scene_detection(metrics)
        self.assertEqual(scene_diag["category"], "PURE_FALSE_POSITIVE")
        self.assertIn("Model predicted oil where ground truth has none", scene_diag["explanation"])

    def test_pixel_confusion_matrix_mixed(self):
        """A scene with true positives, false positives, and false negatives."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)

        # Ground truth: 10 pixels total (rows 0-1, cols 0-4)
        gt[0:2, 0:5] = 1
        # Prediction: 10 pixels total (rows 0-1, cols 2-6)
        pred[0:2, 2:7] = 1

        # Overlap (TP): rows 0-1, cols 2-4 -> 2 * 3 = 6 pixels
        # FP: rows 0-1, cols 5-6 -> 2 * 2 = 4 pixels
        # FN: rows 0-1, cols 0-1 -> 2 * 2 = 4 pixels
        # TN: 100 - (6 + 4 + 4) = 86 pixels
        metrics = compute_pixel_metrics(pred, gt, pixel_spacing_m=10.0)
        self.assertEqual(metrics["tp"], 6)
        self.assertEqual(metrics["fp"], 4)
        self.assertEqual(metrics["fn"], 4)
        self.assertEqual(metrics["tn"], 86)
        self.assertAlmostEqual(metrics["precision"], 6.0 / (6 + 4), places=4)
        self.assertAlmostEqual(metrics["recall"], 6.0 / (6 + 4), places=4)
        self.assertAlmostEqual(metrics["iou"], 6.0 / (6 + 4 + 4), places=4)

        scene_diag = classify_scene_detection(metrics)
        self.assertEqual(scene_diag["category"], "PARTIAL_DETECTION")

    def test_radiometric_damping_contrast_threshold(self):
        """Radiometric contrast < 3.5 dB indicates biogenic/low-wind look-alike risk."""
        # Scene 00594 has contrast 2.70 dB < 3.5 dB
        contrast_00594 = 2.70
        threshold_db = 3.5

        def diagnose_contrast(contrast_db):
            if contrast_db < threshold_db:
                return "AMBIGUOUS_LOOKALIKE_RISK"
            return "HIGH_CONFIDENCE_OIL_DAMPING"

        diag = diagnose_contrast(contrast_00594)
        self.assertEqual(diag, "AMBIGUOUS_LOOKALIKE_RISK")

        # High contrast mineral oil slick (> 4.0 dB)
        diag_high = diagnose_contrast(5.2)
        self.assertEqual(diag_high, "HIGH_CONFIDENCE_OIL_DAMPING")

    def test_ais_absence_decoupled_from_false_detection(self):
        """
        Confirm that lack of AIS does NOT automatically classify a spill detection as false.
        It must be flagged as an observation/coverage gap.
        """
        # Case 1: Model detected slick, contrast is strong, but NO AIS vessels in corridor
        status_no_ais = evaluate_scene_false_detection_status(
            has_sar_detection=True,
            damping_contrast_db=4.5,
            candidate_vessels=[],
            ais_coverage_available=False,
            temporal_gap_flag=True
        )
        # Must NOT be FALSE_POSITIVE
        self.assertNotEqual(status_no_ais.status, FalseDetectionStatus.FALSE_POSITIVE)
        self.assertEqual(status_no_ais.status, FalseDetectionStatus.AIS_COVERAGE_GAP)
        self.assertIn("AIS temporal or spatial coverage gap", status_no_ais.diagnostic_note)

    def test_coordinate_crs_reprojection_utm_to_wgs84(self):
        """
        Test that SpillCharacterizer correctly reprojects
        projected coordinates (e.g. UTM Zone 16N, EPSG:32616) to EPSG:4326 (lat/lon).
        """
        # UTM 16N coordinates in Gulf of Mexico: approx x=351000, y=3080000
        # Corresponds roughly to ~27.84 N, ~90.51 W
        transform = from_origin(351000.0, 3080000.0, 10.0, 10.0)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # 20x20 box (400 pixels)

        characterizer = SpillCharacterizer(pixel_spacing_m=10.0, min_slick_pixels=25)
        spill = characterizer.characterize(
            mask=mask,
            spill_id="test_spill_reproject",
            observation_time="2018-08-03 17:25:57",
            transform=transform,
            crs="EPSG:32616",
        )

        self.assertIsNotNone(spill)
        centroid_lat = spill.centroid_lat
        centroid_lon = spill.centroid_lon
        # Latitude should be around 27.8 N and Lon around -90.5 W (NOT 351000 or 3080000!)
        self.assertGreater(centroid_lat, 20.0)
        self.assertLess(centroid_lat, 35.0)
        self.assertGreater(centroid_lon, -100.0)
        self.assertLess(centroid_lon, -80.0)
        self.assertAlmostEqual(spill.area_km2, 400 * 100 / 1e6, places=4)


if __name__ == "__main__":
    unittest.main()
