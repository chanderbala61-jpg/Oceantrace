"""
Comprehensive Unit Test Suite for OceanTrace Hindcasting Engine
---------------------------------------------------------------
Tests:
  1. DirectObservationBaseline
  2. Coordinate displacement: Northward
  3. Coordinate displacement: Southward
  4. Coordinate displacement: Eastward
  5. Coordinate displacement: Westward
  6. Backward integration logic (retro-stepping)
  7. Longitude meridian wrapping (-180 / +180)
  8. Latitude boundary validation
  9. Meteorological to vector conversion
  10. Zero wind handling
  11. Missing environmental provider fallback
  12. Uncertainty radius expansion over duration
  13. Discrete trajectory history generation
  14. Release time window calculation
  15. GeoJSON export integrity
  16. Plot generation
  17. SpillEvent update bridge
  18. Full End-to-End Spill -> Hindcast -> AIS Pipeline Integration
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from src.ais.schema import SpillEvent
from src.ais.loader import load_ais_csv
from src.ais import run_ais_correlation_pipeline
from src.hindcasting.schema import HindcastResult, EnvironmentalVector
from src.hindcasting.coordinates import displace_lat_lon, haversine_km
from src.hindcasting.environmental import (
    ConstantEnvironmentalProvider,
    meteorological_to_vector
)
from src.hindcasting.models.direct_baseline import DirectObservationBaseline
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.visualization import export_hindcast_to_geojson, plot_hindcast_trajectory
from src.hindcasting import update_spill_event_from_hindcast
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture


class TestHindcastingEngine(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.obs_time = datetime(2026, 9, 4, 12, 0, 0)
        self.spill = SpillEvent(
            spill_id="SPILL_SAR_TEST",
            observation_time=self.obs_time,
            centroid_lat=28.50,
            centroid_lon=-88.50,
            area_km2=8.5,
            origin_uncertainty_km=5.0
        )

    # 1. Direct Observation Baseline
    def test_direct_observation_baseline(self):
        model = DirectObservationBaseline()
        res = model.estimate_origin(self.spill, drift_hours=12.0)
        
        self.assertEqual(res.estimated_origin_lat, 28.50)
        self.assertEqual(res.estimated_origin_lon, -88.50)
        self.assertEqual(res.origin_time_end, self.obs_time)
        self.assertEqual(res.origin_time_start, self.obs_time - timedelta(hours=12))
        self.assertGreaterEqual(res.origin_uncertainty_km, 15.0)
        self.assertIn("unavailable", res.warnings[0].lower())

    # 2. Coordinate displacement: Northward
    def test_displacement_north(self):
        # 111,195 meters North is approximately +1.0 degree latitude
        new_lat, new_lon = displace_lat_lon(0.0, 0.0, dx_meters=0.0, dy_meters=111195.0)
        self.assertAlmostEqual(new_lat, 1.0, places=2)
        self.assertAlmostEqual(new_lon, 0.0, places=4)

    # 3. Coordinate displacement: Southward
    def test_displacement_south(self):
        new_lat, new_lon = displace_lat_lon(10.0, 0.0, dx_meters=0.0, dy_meters=-111195.0)
        self.assertAlmostEqual(new_lat, 9.0, places=2)
        self.assertAlmostEqual(new_lon, 0.0, places=4)

    # 4. Coordinate displacement: Eastward
    def test_displacement_east(self):
        # At equator, 111,195 meters East is approximately +1.0 degree longitude
        new_lat, new_lon = displace_lat_lon(0.0, 0.0, dx_meters=111195.0, dy_meters=0.0)
        self.assertAlmostEqual(new_lat, 0.0, places=4)
        self.assertAlmostEqual(new_lon, 1.0, places=2)

    # 5. Coordinate displacement: Westward
    def test_displacement_west(self):
        new_lat, new_lon = displace_lat_lon(0.0, 0.0, dx_meters=-111195.0, dy_meters=0.0)
        self.assertAlmostEqual(new_lat, 0.0, places=4)
        self.assertAlmostEqual(new_lon, -1.0, places=2)

    # 6. Backward Integration Logic
    def test_backward_integration_direction(self):
        # If wind is blowing TOWARDS East (u = +10 m/s, v = 0),
        # forward drift pushed oil Eastward.
        # Therefore historical origin MUST be WEST of observed slick!
        wind = EnvironmentalVector(u=10.0, v=0.0, source="test_eastward_wind")
        provider = ConstantEnvironmentalProvider(wind=wind)

        model = SimpleWindDriftModel(windage_factor=0.03, time_step_hours=1.0)
        res = model.estimate_origin(self.spill, drift_hours=10.0, env_provider=provider)

        # Origin longitude must be less than observed longitude (further West)
        self.assertLess(res.estimated_origin_lon, self.spill.centroid_lon)
        # Latitude should remain roughly identical
        self.assertAlmostEqual(res.estimated_origin_lat, self.spill.centroid_lat, places=2)

    # 7. Longitude Meridian Wrapping
    def test_longitude_meridian_wrapping(self):
        # Near antimeridian (+179.99), 50km East displacement must wrap to negative longitudes
        new_lat, new_lon = displace_lat_lon(0.0, 179.99, dx_meters=50000.0, dy_meters=0.0)
        self.assertLess(new_lon, 0.0)
        self.assertGreaterEqual(new_lon, -180.0)

    # 8. Latitude boundary validation
    def test_invalid_latitude_validation(self):
        with self.assertRaises(ValueError):
            displace_lat_lon(95.0, 0.0, 0.0, 0.0)

    # 9. Meteorological to vector conversion
    def test_meteorological_conversion(self):
        # Wind FROM North (0 deg / 360 deg) blows TOWARDS South: u = 0, v = -speed
        vec_n = meteorological_to_vector(speed_ms=10.0, direction_from_deg=0.0)
        self.assertAlmostEqual(vec_n.u, 0.0, places=4)
        self.assertAlmostEqual(vec_n.v, -10.0, places=4)

        # Wind FROM West (270 deg) blows TOWARDS East: u = +speed, v = 0
        vec_w = meteorological_to_vector(speed_ms=10.0, direction_from_deg=270.0)
        self.assertAlmostEqual(vec_w.u, 10.0, places=4)
        self.assertAlmostEqual(vec_w.v, 0.0, places=4)

    # 10. Zero wind handling
    def test_zero_wind(self):
        provider = ConstantEnvironmentalProvider(
            wind=EnvironmentalVector(u=0.0, v=0.0)
        )
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=6.0, env_provider=provider)
        
        self.assertAlmostEqual(res.estimated_origin_lat, self.spill.centroid_lat, places=4)
        self.assertAlmostEqual(res.estimated_origin_lon, self.spill.centroid_lon, places=4)

    # 11. Missing environmental provider fallback
    def test_missing_provider_fallback(self):
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=6.0, env_provider=None)
        
        self.assertGreater(len(res.warnings), 0)
        self.assertAlmostEqual(res.estimated_origin_lat, self.spill.centroid_lat, places=4)

    # 12. Uncertainty radius expansion
    def test_uncertainty_expansion(self):
        model = SimpleWindDriftModel(base_uncertainty_km=5.0, uncertainty_rate_km_per_hr=0.5)
        res_6h = model.estimate_origin(self.spill, drift_hours=6.0)
        res_24h = model.estimate_origin(self.spill, drift_hours=24.0)
        
        # 5.0 + 6*0.5 = 8.0 km
        self.assertAlmostEqual(res_6h.origin_uncertainty_km, 8.0, places=2)
        # 5.0 + 24*0.5 = 17.0 km
        self.assertAlmostEqual(res_24h.origin_uncertainty_km, 17.0, places=2)
        self.assertGreater(res_24h.origin_uncertainty_km, res_6h.origin_uncertainty_km)

    # 13. Discrete Trajectory Points
    def test_discrete_trajectory_generation(self):
        wind = EnvironmentalVector(u=5.0, v=-5.0)
        provider = ConstantEnvironmentalProvider(wind=wind)
        model = SimpleWindDriftModel(time_step_hours=1.0)
        res = model.estimate_origin(self.spill, drift_hours=5.0, env_provider=provider)
        
        # t=0, -1, -2, -3, -4, -5 -> 6 discrete points
        self.assertEqual(len(res.trajectory), 6)
        self.assertEqual(res.trajectory[0].step_hours_from_obs, 0.0)
        self.assertEqual(res.trajectory[-1].step_hours_from_obs, -5.0)

    # 14. Origin time window
    def test_origin_time_window(self):
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=12.0)
        
        self.assertEqual(res.origin_time_end, self.obs_time)
        self.assertEqual(res.origin_time_start, self.obs_time - timedelta(hours=12.0))

    # 15. GeoJSON export integrity
    def test_geojson_export(self):
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=6.0)
        out_path = os.path.join(self.tmp_dir, "hindcast.geojson")
        export_hindcast_to_geojson(res, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 200)

    # 16. Plot rendering
    def test_plot_rendering(self):
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=6.0)
        out_png = os.path.join(self.tmp_dir, "hindcast.png")
        plot_hindcast_trajectory(res, out_png)
        
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)

    # 17. SpillEvent update bridge
    def test_update_spill_event_bridge(self):
        model = SimpleWindDriftModel()
        res = model.estimate_origin(self.spill, drift_hours=8.0)
        
        updated_spill = update_spill_event_from_hindcast(self.spill, res)
        self.assertEqual(updated_spill.estimated_origin_lat, res.estimated_origin_lat)
        self.assertEqual(updated_spill.estimated_origin_lon, res.estimated_origin_lon)
        self.assertEqual(updated_spill.origin_uncertainty_km, res.origin_uncertainty_km)
        self.assertTrue(updated_spill.has_hindcast_origin)

    # 18. Full End-to-End Pipeline: Spill Detection -> Hindcasting -> AIS Correlation
    def test_full_pipeline_spill_to_hindcast_to_ais(self):
        # Step A: Synthetic AIS dataset
        ais_csv = os.path.join(self.tmp_dir, "pipeline_ais.csv")
        create_synthetic_ais_fixture(ais_csv)

        # Step B: Spill detected at (28.52, -88.48)
        raw_spill = SpillEvent(
            spill_id="E2E_SPILL_001",
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            centroid_lat=28.52,
            centroid_lon=-88.48,
            area_km2=10.0,
            origin_uncertainty_km=5.0
        )
        self.assertIsNone(raw_spill.estimated_origin_lat)

        # Step C: Hindcasting - Slick was pushed Northeast by Southwesterly wind (u=+5, v=+5)
        # Backward simulation will shift origin Southwest back towards (28.50, -88.50)
        wind = EnvironmentalVector(u=5.14, v=5.14, source="SYNTHETIC_TEST_FIXTURE_ONLY")
        env_prov = ConstantEnvironmentalProvider(wind=wind)
        hindcast_engine = SimpleWindDriftModel(windage_factor=0.03, time_step_hours=1.0)
        
        hindcast_res = hindcast_engine.estimate_origin(raw_spill, drift_hours=2.0, env_provider=env_prov)
        self.assertAlmostEqual(hindcast_res.estimated_origin_lat, 28.50, delta=0.02)
        self.assertAlmostEqual(hindcast_res.estimated_origin_lon, -88.50, delta=0.02)

        # Step D: Bridge to SpillEvent
        ready_spill = update_spill_event_from_hindcast(raw_spill, hindcast_res)
        self.assertTrue(ready_spill.has_hindcast_origin)

        # Step E: AIS Correlation using hindcasted origin
        ais_output = run_ais_correlation_pipeline(
            spill=ready_spill,
            ais_csv_path=ais_csv,
            lookback_hours=12.0,
            lookforward_hours=6.0,
            search_radius_km=50.0
        )

        ranked = ais_output["ranked_candidates"]
        self.assertGreater(len(ranked), 0)
        # Top candidate must be MMSI 111222333 with high compatibility
        self.assertEqual(ranked[0].mmsi, 111222333)
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[0].classification, "Highly Compatible Candidate")


if __name__ == "__main__":
    unittest.main()
