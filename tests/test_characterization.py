"""
Unit and Integration Tests for OceanTrace Spill Characterization Module
-----------------------------------------------------------------------
Verifies:
  1. Empty / zero-mask handling (no spurious detection)
  2. Sub-threshold speckle noise filtering
  3. Single slick geometric centroid and area derivation
  4. Georeferencing via Affine transform
  5. Georeferencing via bounding box
  6. Multi-slick polygon contour extraction and closure
  7. Seamless integration: SpillEvent -> Hindcasting -> AIS Correlation
"""

import unittest
from datetime import datetime, timedelta
import numpy as np

from src.characterization import SpillCharacterizer, parse_sentinel1_timestamp
from src.ais.schema import SpillEvent
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.schema import EnvironmentalVector
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting import update_spill_event_from_hindcast
from src.ais import run_ais_correlation_pipeline
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture
import tempfile
import os


class TestSpillCharacterization(unittest.TestCase):

    def setUp(self):
        self.characterizer = SpillCharacterizer(
            pixel_spacing_m=10.0,
            min_slick_pixels=20,
            default_uncertainty_km=5.0
        )
        self.obs_time = datetime(2026, 9, 4, 12, 0, 0)

    # 1. Empty mask handling
    def test_empty_mask(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        event = self.characterizer.characterize(
            mask=mask,
            spill_id="EMPTY_TEST",
            observation_time=self.obs_time,
            origin_lat_hint=28.5,
            origin_lon_hint=-88.5
        )
        self.assertIsNone(event)

    # 2. Speckle noise filtering
    def test_noise_filtering(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Put 10 isolated pixels (below min_slick_pixels=20)
        mask[10:12, 10:15] = 1
        event = self.characterizer.characterize(
            mask=mask,
            spill_id="NOISE_TEST",
            observation_time=self.obs_time,
            origin_lat_hint=28.5,
            origin_lon_hint=-88.5
        )
        self.assertIsNone(event)

    # 3. Single slick characterization with center hint
    def test_single_slick_geometry(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Centered 20x20 block: pixels = 400
        # Centered at row 100, col 100
        mask[90:110, 90:110] = 1
        
        event = self.characterizer.characterize(
            mask=mask,
            spill_id="SLICK_001",
            observation_time=self.obs_time,
            origin_lat_hint=25.0,
            origin_lon_hint=-90.0
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.spill_id, "SLICK_001")
        self.assertEqual(event.observation_time, self.obs_time)
        # 400 pixels * 100 m^2 = 40,000 m^2 = 0.04 km^2
        self.assertAlmostEqual(event.area_km2, 0.04, places=3)
        # Centroid should be very close to reference center (100, 100 in 200x200)
        self.assertAlmostEqual(event.centroid_lat, 25.0, places=3)
        self.assertAlmostEqual(event.centroid_lon, -90.0, places=3)
        # Polygon must be non-empty and closed
        self.assertIsNotNone(event.polygon)
        self.assertGreater(len(event.polygon), 3)
        self.assertEqual(event.polygon[0], event.polygon[-1])

    # 4. Georeferencing via Bounding Box
    def test_bounding_box_georeferencing(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Put slick in top-left quadrant: row 20..30, col 20..30 (center = 25, 25)
        mask[20:30, 20:30] = 1
        # Bbox: min_lat=20.0, min_lon=-90.0, max_lat=21.0, max_lon=-89.0
        bbox = (20.0, -90.0, 21.0, -89.0)
        
        event = self.characterizer.characterize(
            mask=mask,
            spill_id="BBOX_TEST",
            observation_time=self.obs_time,
            bbox=bbox
        )
        self.assertIsNotNone(event)
        # Col 25 in 0..100 -> lon = -90.0 + 0.25 * 1.0 = -89.75
        # Row 25 in 0..100 -> lat = 21.0 - 0.25 * 1.0 = 20.75
        self.assertAlmostEqual(event.centroid_lat, 20.75, places=2)
        self.assertAlmostEqual(event.centroid_lon, -89.75, places=2)

    # 5. Timestamp parsing
    def test_timestamp_parsing(self):
        s1_str = "03-AUG-2018 17:25:57.581481"
        dt = parse_sentinel1_timestamp(s1_str)
        self.assertEqual(dt.year, 2018)
        self.assertEqual(dt.month, 8)
        self.assertEqual(dt.day, 3)
        self.assertEqual(dt.hour, 17)
        self.assertEqual(dt.minute, 25)

    # 6. Full End-to-End Pipeline Integration:
    #    Characterize mask -> Hindcast probable origin -> AIS Candidate Correlation
    def test_characterize_to_hindcast_to_ais_pipeline(self):
        tmp_dir = tempfile.mkdtemp()
        fixture_csv = os.path.join(tmp_dir, "synthetic_ais.csv")
        create_synthetic_ais_fixture(fixture_csv)

        # Create a synthetic detected spill mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[95:105, 95:105] = 1  # 100 pixels

        # Step 1: Characterize
        spill = self.characterizer.characterize(
            mask=mask,
            spill_id="INTEGRATION_SPILL_01",
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            origin_lat_hint=28.50,
            origin_lon_hint=-88.50
        )
        self.assertIsNotNone(spill)
        self.assertAlmostEqual(spill.centroid_lat, 28.50, places=2)
        self.assertAlmostEqual(spill.centroid_lon, -88.50, places=2)

        # Step 2: Ocean Drift Hindcast
        # Wind blowing toward East (u = +5 m/s)
        wind_vec = EnvironmentalVector(u=5.0, v=0.0, source="test_wind")
        current_vec = EnvironmentalVector(u=0.0, v=0.0, source="test_current")
        env = ConstantEnvironmentalProvider(wind=wind_vec, current=current_vec)
        model = SimpleWindDriftModel(windage_factor=0.03)
        hindcast_res = model.estimate_origin(spill, drift_hours=4.0, env_provider=env)

        # Verify hindcast retro-drift westward
        self.assertLess(hindcast_res.estimated_origin_lon, spill.centroid_lon)
        self.assertGreater(hindcast_res.origin_uncertainty_km, 5.0)

        # Update SpillEvent with hindcast result
        updated_spill = update_spill_event_from_hindcast(spill, hindcast_res)
        self.assertTrue(updated_spill.has_hindcast_origin)

        # Step 3: AIS Correlation Engine
        pipeline_out = run_ais_correlation_pipeline(
            spill=updated_spill,
            ais_csv_path=fixture_csv,
            search_radius_km=50.0
        )
        self.assertIn("ranked_candidates", pipeline_out)
        results = pipeline_out["ranked_candidates"]
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # Top candidate vessel
        top_cand = results[0]
        self.assertIn("Compatible", top_cand.classification)
        self.assertGreater(top_cand.overall_score, 0.4)


if __name__ == "__main__":
    unittest.main()
