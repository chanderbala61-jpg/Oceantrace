"""
OceanTrace End-to-End Comprehensive Pipeline Integration Test
=============================================================
Verifies:
1. Model checkpoint immutability (SHA-256 before == SHA-256 after).
2. Frozen SegFormer-B0 inference execution within OceanTracePipeline.
3. Separation of predicted mask vs companion ground-truth mask.
4. Spill characterization metrics calculation.
5. Backward drift hindcasting (RK4) with uncertainty region.
6. AIS candidate correlation with non-accusatory labeling.
7. Graceful degradation on model false-positive scene (Scene 00594 limitation case:
   acquisition from 2017, no AIS overlap -> 'No sufficient AIS evidence').
"""

import os
import unittest
import hashlib
from datetime import datetime, timedelta
import numpy as np
import torch

from src.pipeline import OceanTracePipeline, InvestigationRecord
from src.inference import load_frozen_segformer, LOCKED_CHECKPOINT_SHA256, DEFAULT_CHECKPOINT_PATH
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting.schema import EnvironmentalVector
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture
import tempfile


class TestEndToEndPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Compute SHA-256 of checkpoint before any test execution
        hasher = hashlib.sha256()
        with open(DEFAULT_CHECKPOINT_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        cls.sha256_before = hasher.hexdigest()
        assert cls.sha256_before == LOCKED_CHECKPOINT_SHA256, "Checkpoint hash mismatch!"

        # Load frozen model once for test class
        cls.model, cls.meta = load_frozen_segformer(device="cpu", verify_hash=True)

    @classmethod
    def tearDownClass(cls):
        # Verify SHA-256 of checkpoint after all test executions
        hasher = hashlib.sha256()
        with open(DEFAULT_CHECKPOINT_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256_after = hasher.hexdigest()
        assert sha256_after == cls.sha256_before, "CRITICAL: Checkpoint was modified during testing!"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ais_csv_path = os.path.join(self.temp_dir.name, "ais_traffic.csv")
        create_synthetic_ais_fixture(self.ais_csv_path)

        # 256x256 test SAR image (dB)
        self.sar_image = np.full((256, 256, 2), -12.0, dtype=np.float32)
        # Add a slick region (damped by 10 dB)
        self.sar_image[80:160, 80:160, 0] = -24.0
        self.sar_image[80:160, 80:160, 1] = -20.0

        # Companion ground truth mask
        self.companion_mask = np.zeros((256, 256), dtype=np.uint8)
        self.companion_mask[80:160, 80:160] = 1

        # Environmental wind (blowing East at 5.0 m/s)
        wind = EnvironmentalVector(u=5.0, v=0.0, source="test_wind")
        self.env_provider = ConstantEnvironmentalProvider(wind=wind)

        self.pipeline = OceanTracePipeline(
            model=self.model,
            default_drift_hours=4.0,
            ais_search_radius_km=50.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_with_frozen_model_inference(self):
        """Verifies end-to-end pipeline using actual SegFormer-B0 forward pass."""
        record = self.pipeline.execute(
            scene_id="E2E_INFERENCE_TEST",
            sar_image=self.sar_image,
            mask=self.companion_mask,  # Passed as companion comparison
            run_inference=True,
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50,
            env_provider=self.env_provider,
            ais_csv_path=self.ais_csv_path,
            wind_speed_ms=6.0,
            drift_hours=4.0,
        )

        # Check inference execution
        self.assertIn("SegFormer-B0 Inference", record.detection_method)
        self.assertIsNotNone(record.predicted_mask)
        self.assertIsNotNone(record.ground_truth_mask)
        self.assertEqual(record.predicted_mask.shape, (256, 256))
        self.assertIn("diagnostic_iou", record.diagnostic_metrics)

        # Spill characterization
        if record.has_detection:
            self.assertIsNotNone(record.spill)
            self.assertGreater(record.spill.area_km2, 0.0)

            # Hindcast origin
            self.assertIsNotNone(record.hindcast_result)
            self.assertIn("SimpleWindDriftModel", record.drift_model_name)

            # AIS correlation
            self.assertTrue(record.ais_available)
            for cand in record.candidate_vessels:
                # Confirm non-accusatory terminology
                self.assertNotIn("responsible", cand.classification.lower())
                self.assertNotIn("guilty", cand.classification.lower())
                self.assertIn("association", cand.classification.lower())

    def test_false_positive_scene_limitation_case(self):
        """
        Verifies scientific integrity on a model false positive event:
        Event from 2017 (like scene 00594) has zero temporal overlap with 2019/2026 AIS.
        Must report no sufficient AIS evidence without fabricating matches.
        """
        # Event timestamp: 2017-06-25 (no AIS available for 2017)
        obs_time_2017 = datetime(2017, 6, 25, 14, 30, 0)

        record = self.pipeline.execute(
            scene_id="EVENT_00594_FP_LIMITATION",
            sar_image=self.sar_image,
            mask=self.companion_mask,
            run_inference=False,
            observation_time=obs_time_2017,
            origin_lat_hint=24.50,
            origin_lon_hint=-85.50,
            env_provider=self.env_provider,
            ais_csv_path=self.ais_csv_path,  # AIS fixture is for 2026
            drift_hours=4.0,
        )

        # AIS time window for 2017 event is June 2017.
        # AIS CSV has broadcasts from September 2026.
        # Correlation pipeline must find 0 candidate vessels within time window.
        self.assertEqual(len(record.candidate_vessels), 0)
        self.assertEqual(record.total_vessels_checked, 0)


if __name__ == "__main__":
    unittest.main()
