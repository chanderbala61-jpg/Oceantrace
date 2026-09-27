"""
Unit and Integration Tests for OceanTrace End-to-End Pipeline & Dashboard
=========================================================================
Verifies:
  1. Full End-to-End Pipeline execution (Detect -> Characterize -> Trace -> Correlate -> Output).
  2. Look-Alike & Radiometric Ambiguity integration.
  3. Missing AIS handling (graceful degradation, no fabrication).
  4. Missing Environmental data handling (verified fallback).
  5. Negative detection handling (empty masks).
  6. Dashboard visual rendering & standalone HTML report generation.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
import numpy as np

from src.pipeline import OceanTracePipeline, InvestigationRecord
from src.characterization import SpillCharacterizer
from src.look_alike import LookAlikeAnalyzer
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting.schema import EnvironmentalVector
from src.dashboard import render_investigation_dashboard, generate_html_investigation_report
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture


class TestOceanTracePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ais_csv_path = os.path.join(self.temp_dir.name, "ais_traffic.csv")
        create_synthetic_ais_fixture(self.ais_csv_path)

        # Synthetic SAR backscatter (200x200, 2 channels)
        # Background sea: -12 dB (VH), -8 dB (VV)
        self.sar_image = np.full((200, 200, 2), -10.0, dtype=np.float32)
        self.sar_image[..., 0] = -14.0
        self.sar_image[..., 1] = -9.0

        # Synthetic oil slick mask: 30x40 patch centered at (100, 100)
        self.mask = np.zeros((200, 200), dtype=np.uint8)
        self.mask[85:115, 80:120] = 1

        # Inside slick: damp by 9 dB -> -23 dB (VH), -18 dB (VV)
        self.sar_image[85:115, 80:120, 0] = -23.0
        self.sar_image[85:115, 80:120, 1] = -18.0

        # Environmental wind blowing toward East (u = +5.0 m/s)
        wind = EnvironmentalVector(u=5.0, v=0.0, source="test_wind")
        self.env_provider = ConstantEnvironmentalProvider(wind=wind)

        self.pipeline = OceanTracePipeline(
            default_drift_hours=4.0,
            ais_search_radius_km=50.0
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_with_ais_correlation(self):
        """Tests complete pipeline run with SAR image, mask, wind, and AIS."""
        record = self.pipeline.execute(
            scene_id="TEST_SCENE_001",
            sar_image=self.sar_image,
            mask=self.mask,
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50,
            env_provider=self.env_provider,
            ais_csv_path=self.ais_csv_path,
            wind_speed_ms=7.0,
            drift_hours=4.0
        )

        # 1. Detection & Characterization
        self.assertTrue(record.has_detection)
        self.assertIsNotNone(record.spill)
        spill = record.spill
        self.assertAlmostEqual(spill.centroid_lat, 28.50, places=2)
        self.assertAlmostEqual(spill.centroid_lon, -88.50, places=2)
        self.assertGreater(spill.area_km2, 0.0)

        # 2. Look-Alike Assessment
        self.assertIsNotNone(record.look_alike_assessment)
        la = record.look_alike_assessment
        self.assertGreater(la.damping_contrast_db, 6.0)
        self.assertFalse(la.is_look_alike_suspect)
        self.assertIn("True Oil Spill", la.classification)

        # 3. Hindcasting
        self.assertIsNotNone(record.hindcast_result)
        h = record.hindcast_result
        # With eastward wind, retro-drift must be westward (< -88.50)
        self.assertLess(h.estimated_origin_lon, spill.centroid_lon)
        self.assertEqual(len(h.trajectory), 5)  # t=0, 1, 2, 3, 4h

        # 4. AIS Candidate Correlation
        self.assertTrue(record.ais_available)
        self.assertGreater(len(record.candidate_vessels), 0)
        top_cand = record.candidate_vessels[0]
        self.assertIn("Compatible", top_cand.classification)
        self.assertGreater(top_cand.overall_score, 0.4)

        # 5. Diagnostic Summary
        summary_text = record.summary()
        self.assertIn("OCEANTRACE INVESTIGATION: TEST_SCENE_001", summary_text)
        self.assertIn("Spill Location (Observed)", summary_text)

    def test_pipeline_missing_ais_data(self):
        """Ensures missing AIS data causes graceful degradation without crashing or fabricating."""
        record = self.pipeline.execute(
            scene_id="TEST_SCENE_NO_AIS",
            sar_image=self.sar_image,
            mask=self.mask,
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50,
            env_provider=self.env_provider,
            ais_csv_path=None,  # No AIS
            drift_hours=4.0
        )

        self.assertTrue(record.has_detection)
        self.assertIsNotNone(record.spill)
        self.assertIsNotNone(record.hindcast_result)
        self.assertFalse(record.ais_available)
        self.assertEqual(len(record.candidate_vessels), 0)
        self.assertTrue(record.missing_data_flags.get("ais_data_missing", False))
        self.assertTrue(any("AIS data unavailable" in w for w in record.warnings))

    def test_pipeline_missing_environmental_data(self):
        """Ensures missing wind/current data triggers documented fallback and warnings."""
        record = self.pipeline.execute(
            scene_id="TEST_SCENE_NO_ENV",
            sar_image=self.sar_image,
            mask=self.mask,
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50,
            env_provider=None,  # No environmental data
            ais_csv_path=self.ais_csv_path,
            drift_hours=4.0
        )

        self.assertTrue(record.has_detection)
        self.assertIsNotNone(record.hindcast_result)
        # Warning about fallback zero drift
        self.assertTrue(any("No environmental data" in w for w in record.warnings))

    def test_pipeline_no_detection(self):
        """Empty mask must cleanly abort downstream processing."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        record = self.pipeline.execute(
            scene_id="TEST_SCENE_EMPTY",
            sar_image=None,
            mask=empty_mask,
        )
        self.assertFalse(record.has_detection)
        self.assertIsNone(record.spill)
        self.assertIsNone(record.hindcast_result)
        self.assertFalse(record.ais_available)

    def test_dashboard_and_html_generation(self):
        """Tests dashboard PNG rendering and HTML report export."""
        record = self.pipeline.execute(
            scene_id="TEST_SCENE_DASHBOARD",
            sar_image=self.sar_image,
            mask=self.mask,
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50,
            env_provider=self.env_provider,
            ais_csv_path=self.ais_csv_path,
            drift_hours=4.0
        )

        png_out = os.path.join(self.temp_dir.name, "dashboard.png")
        fig = render_investigation_dashboard(
            record=record,
            sar_image=self.sar_image,
            mask=self.mask,
            output_path=png_out,
            dpi=100
        )
        self.assertTrue(os.path.isfile(png_out))
        self.assertGreater(os.path.getsize(png_out), 10000)

        html_out = os.path.join(self.temp_dir.name, "report.html")
        html_str = generate_html_investigation_report(
            record=record,
            dashboard_image_relpath="dashboard.png",
            output_html_path=html_out
        )
        self.assertTrue(os.path.isfile(html_out))
        self.assertIn("OceanTrace Maritime Incident Investigation", html_str)
        self.assertIn("TEST_SCENE_DASHBOARD", html_str)
        self.assertIn("Scientific Integrity & Legal Standard", html_str)


if __name__ == "__main__":
    unittest.main()
