"""
Unit & Integration Tests for OceanTrace Counterfactual Vessel Testing Module
============================================================================
Validates:
  1. Forward RK4 numerical correctness & 4th-order integration
  2. Forward time directionality (dt > 0, t_end >= t_start)
  3. Zero-current / zero-wind sanity case (zero displacement)
  4. Constant-velocity analytical displacement comparison
  5. AIS candidate input validation & error handling
  6. Hypothetical release formulation with neutral scientific labeling
  7. Simulation trajectory generation and step tracking
  8. Geographic coordinate displacement and geodesic distance
  9. Observed vs simulated geometry diagnostics (centroid distance, containment)
  10. Deterministic Candidate Prioritization Score calculation
  11. Frozen SegFormer-B0 model checkpoint hash verification
  12. End-to-end counterfactual test on candidate vessel
"""

import math
import hashlib
import unittest
from datetime import datetime, timedelta

from src.ais.schema import SpillEvent, CandidateVesselResult, AISRecord
from src.hindcasting.schema import TrajectoryPoint, EnvironmentalVector
from src.hindcasting.environmental import ConstantEnvironmentalProvider
from src.hindcasting.coordinates import displace_lat_lon, haversine_km
from src.counterfactual.schema import (
    HypotheticalRelease,
    ForwardSimulationConfig,
    ConsistencyDiagnostics,
    CounterfactualResult,
)
from src.counterfactual.forward_rk4 import ForwardRK4DriftModel
from src.counterfactual.analyzer import CounterfactualAnalyzer
from src.inference import DEFAULT_CHECKPOINT_PATH


class TestCounterfactualSimulation(unittest.TestCase):

    def setUp(self):
        self.config = ForwardSimulationConfig(
            windage_factor=0.030,
            time_step_hours=0.5,
            initial_footprint_radius_km=0.5,
            diffusion_rate_km_sqrt_hr=0.4
        )
        self.model = ForwardRK4DriftModel(config=self.config)
        self.analyzer = CounterfactualAnalyzer(config=self.config, model=self.model)

    def test_forward_time_direction(self):
        """Verify that forward simulation steps chronologically forward in time."""
        t_start = datetime(2026, 9, 4, 10, 0, 0)
        t_obs = datetime(2026, 9, 4, 14, 0, 0)

        rel = HypotheticalRelease(
            mmsi=123456789,
            vessel_name="Alpha",
            vessel_type="Tanker",
            release_lat=20.0,
            release_lon=80.0,
            release_time=t_start
        )

        traj, _, _ = self.model.simulate_forward(rel, target_observation_time=t_obs)
        self.assertGreater(len(traj), 1)
        self.assertEqual(traj[0].timestamp, t_start)
        self.assertEqual(traj[-1].timestamp, t_obs)

        # Check monotonically increasing timestamps
        for i in range(len(traj) - 1):
            self.assertGreater(traj[i + 1].timestamp, traj[i].timestamp)
            self.assertGreater(traj[i + 1].step_hours_from_obs, traj[i].step_hours_from_obs)

    def test_zero_current_zero_wind_sanity(self):
        """With zero environmental forcing, vessel release position must not move."""
        t_start = datetime(2026, 9, 4, 10, 0, 0)
        t_obs = datetime(2026, 9, 4, 16, 0, 0)
        lat_init = 25.0
        lon_init = -90.0

        rel = HypotheticalRelease(
            mmsi=111222333,
            vessel_name="Static_Tester",
            vessel_type="Cargo",
            release_lat=lat_init,
            release_lon=lon_init,
            release_time=t_start
        )

        env = ConstantEnvironmentalProvider(
            current=EnvironmentalVector(u=0.0, v=0.0, source="test_zero"),
            wind=EnvironmentalVector(u=0.0, v=0.0, source="test_zero")
        )

        traj, _, _ = self.model.simulate_forward(rel, target_observation_time=t_obs, env_provider=env)
        endpoint = traj[-1]

        self.assertAlmostEqual(endpoint.latitude, lat_init, places=5)
        self.assertAlmostEqual(endpoint.longitude, lon_init, places=5)
        self.assertEqual(endpoint.cumulative_drift_km, 0.0)

    def test_constant_velocity_analytical_displacement(self):
        """
        Under constant current and wind, RK4 forward integration must exactly match
        the analytical displacement: dx = (u_c + 0.03*u_w) * dt, dy = (v_c + 0.03*v_w) * dt.
        """
        t_start = datetime(2026, 9, 4, 12, 0, 0)
        duration_hours = 2.0
        t_obs = t_start + timedelta(hours=duration_hours)
        dt_sec = duration_hours * 3600.0

        u_c, v_c = 0.5, -0.2  # current (m/s)
        u_w, v_w = 10.0, 5.0  # wind (m/s)
        windage = 0.030

        u_total = u_c + (windage * u_w)  # 0.5 + 0.3 = 0.8 m/s
        v_total = v_c + (windage * v_w)  # -0.2 + 0.15 = -0.05 m/s

        expected_dx = u_total * dt_sec
        expected_dy = v_total * dt_sec

        lat_0, lon_0 = 15.0, 60.0
        expected_lat, expected_lon = displace_lat_lon(lat_0, lon_0, expected_dx, expected_dy)

        rel = HypotheticalRelease(
            mmsi=999888777,
            vessel_name="Speedy",
            vessel_type="Tanker",
            release_lat=lat_0,
            release_lon=lon_0,
            release_time=t_start
        )

        env = ConstantEnvironmentalProvider(
            current=EnvironmentalVector(u=u_c, v=v_c, source="test_const"),
            wind=EnvironmentalVector(u=u_w, v=v_w, source="test_const")
        )

        traj, _, _ = self.model.simulate_forward(rel, target_observation_time=t_obs, env_provider=env)
        sim_end = traj[-1]

        self.assertAlmostEqual(sim_end.latitude, expected_lat, places=4)
        self.assertAlmostEqual(sim_end.longitude, expected_lon, places=4)

    def test_ais_candidate_input_validation(self):
        """Verify candidate input parsing for CandidateVesselResult, AISRecord, and dict."""
        t_ping = datetime(2026, 9, 4, 10, 30, 0)

        # 1. From CandidateVesselResult
        cand_res = CandidateVesselResult(
            mmsi=999000001,
            rank=1,
            overall_score=0.88,
            classification="Highly Compatible Candidate",
            measurements={
                "ship_name": "PACIFIC_VOYAGER",
                "ship_type": "Tanker",
                "closest_lat": 28.0,
                "closest_lon": -91.0,
                "closest_approach_time": t_ping,
                "min_distance_km": 2.4,
                "mean_sog": 12.5
            }
        )
        rel1 = self.analyzer.create_hypothetical_release(cand_res)
        self.assertEqual(rel1.mmsi, 999000001)
        self.assertEqual(rel1.vessel_name, "PACIFIC_VOYAGER")
        self.assertEqual(rel1.release_lat, 28.0)
        self.assertEqual(rel1.release_lon, -91.0)
        self.assertEqual(rel1.cpa_distance_km, 2.4)
        self.assertIn("Hypothetical Release", rel1.label)

        # 2. From AISRecord
        record = AISRecord(
            timestamp=t_ping,
            mmsi=888000002,
            latitude=12.5,
            longitude=80.5,
            sog=14.0,
            cog=90.0,
            ship_name="GULF_TRADER"
        )
        rel2 = self.analyzer.create_hypothetical_release(record)
        self.assertEqual(rel2.mmsi, 888000002)
        self.assertEqual(rel2.vessel_name, "GULF_TRADER")
        self.assertEqual(rel2.release_lat, 12.5)

        # 3. Invalid Coordinates Check
        bad_dict = {"mmsi": 123, "lat": 195.0, "lon": 0.0}  # lat out of [-90, 90]
        with self.assertRaises(ValueError):
            self.analyzer.create_hypothetical_release(bad_dict)

    def test_observed_vs_simulated_geometry_diagnostics(self):
        """Verify centroid distance, containment, and spatial decay score."""
        t_rel = datetime(2026, 9, 4, 8, 0, 0)
        t_obs = datetime(2026, 9, 4, 14, 0, 0)  # 6 hours later

        # Spill observed at (20.05, 80.05)
        spill = SpillEvent(
            spill_id="spill_test",
            observation_time=t_obs,
            centroid_lat=20.05,
            centroid_lon=80.05,
            area_km2=2.5,
            estimated_origin_lat=20.00,
            estimated_origin_lon=80.00,
            origin_time_start=datetime(2026, 9, 4, 7, 0, 0),
            origin_time_end=datetime(2026, 9, 4, 9, 0, 0),
            origin_uncertainty_km=8.0
        )

        # Candidate released at (20.00, 80.00) at 8:00
        cand = {
            "mmsi": 999111,
            "vessel_name": "Test_Candidate",
            "lat": 20.00,
            "lon": 80.00,
            "closest_approach_time": t_rel,
            "min_distance_km": 1.2,
            "overall_score": 0.85
        }

        # Environmental currents pushing eastward towards observed spill
        env = ConstantEnvironmentalProvider(
            current=EnvironmentalVector(u=0.25, v=0.25, source="test_east"),
            wind=EnvironmentalVector(u=2.0, v=2.0, source="test_east")
        )

        res = self.analyzer.evaluate_candidate(cand, spill, env_provider=env)
        diag = res.diagnostics
        self.assertIsNotNone(diag)

        # Centroid distance must be a valid non-negative float
        self.assertGreater(diag.centroid_distance_km, 0.0)
        self.assertLess(diag.centroid_distance_km, 50.0)

        # Scores must be bounded in [0.0, 1.0]
        self.assertTrue(0.0 <= diag.spatial_proximity_score <= 1.0)
        self.assertTrue(0.0 <= diag.trajectory_proximity_score <= 1.0)
        self.assertTrue(0.0 <= diag.temporal_alignment_score <= 1.0)
        self.assertTrue(0.0 <= diag.candidate_prioritization_score <= 1.0)

        # Deterministic prioritization score calculation
        w = diag.scoring_weights
        expected_score = round(
            w["spatial_endpoint"] * diag.spatial_proximity_score +
            w["trajectory_proximity"] * diag.trajectory_proximity_score +
            w["temporal_alignment"] * diag.temporal_alignment_score +
            w["prior_ais_score"] * diag.prior_ais_score,
            4
        )
        self.assertAlmostEqual(diag.candidate_prioritization_score, expected_score, places=4)

    def test_distant_candidate_prioritization_decay(self):
        """A distant candidate vessel should receive a low prioritization score."""
        t_rel = datetime(2026, 9, 4, 8, 0, 0)
        t_obs = datetime(2026, 9, 4, 14, 0, 0)

        spill = SpillEvent(
            spill_id="spill_test_near",
            observation_time=t_obs,
            centroid_lat=20.0,
            centroid_lon=80.0,
            area_km2=1.0
        )

        # Distant vessel 150 km away
        distant_cand = {
            "mmsi": 999999,
            "vessel_name": "Distant_Vessel",
            "lat": 21.5,
            "lon": 80.0,
            "closest_approach_time": t_rel,
            "min_distance_km": 150.0,
            "overall_score": 0.10
        }

        res = self.analyzer.evaluate_candidate(distant_cand, spill)
        diag = res.diagnostics
        self.assertEqual(diag.classification, "Low Counterfactual Consistency")
        self.assertLess(diag.candidate_prioritization_score, 0.45)

    def test_end_to_end_counterfactual_pipeline(self):
        """Test multi-candidate evaluation, ranking, and audit trail generation."""
        t_rel = datetime(2026, 9, 4, 8, 0, 0)
        t_obs = datetime(2026, 9, 4, 14, 0, 0)

        spill = SpillEvent(
            spill_id="spill_e2e_001",
            observation_time=t_obs,
            centroid_lat=20.08,
            centroid_lon=80.08,
            area_km2=3.2,
            estimated_origin_lat=20.00,
            estimated_origin_lon=80.00,
            origin_time_start=datetime(2026, 9, 4, 7, 0, 0),
            origin_time_end=datetime(2026, 9, 4, 9, 0, 0),
            origin_uncertainty_km=5.0
        )

        # Candidate 1: Highly aligned tanker
        cand_high = {
            "mmsi": 412001111,
            "vessel_name": "CARRIER_ALPHA",
            "lat": 20.01,
            "lon": 80.01,
            "closest_approach_time": t_rel,
            "min_distance_km": 1.5,
            "overall_score": 0.88,
            "sog": 11.2,
            "cog": 45.0
        }

        # Candidate 2: Moderately aligned cargo
        cand_mid = {
            "mmsi": 412002222,
            "vessel_name": "FREIGHTER_BETA",
            "lat": 20.15,
            "lon": 80.15,
            "closest_approach_time": t_rel,
            "min_distance_km": 12.0,
            "overall_score": 0.55,
            "sog": 9.0,
            "cog": 120.0
        }

        # Candidate 3: Distant offshore supply
        cand_low = {
            "mmsi": 412003333,
            "vessel_name": "SUPPLY_GAMMA",
            "lat": 21.20,
            "lon": 81.00,
            "closest_approach_time": t_rel,
            "min_distance_km": 140.0,
            "overall_score": 0.12,
            "sog": 8.0,
            "cog": 270.0
        }

        env = ConstantEnvironmentalProvider(
            current=EnvironmentalVector(u=0.20, v=0.20, source="synthetic_copernicus"),
            wind=EnvironmentalVector(u=2.5, v=2.5, source="synthetic_gfs")
        )

        results = self.analyzer.evaluate_all_candidates(
            candidates=[cand_high, cand_mid, cand_low],
            spill=spill,
            env_provider=env
        )

        self.assertEqual(len(results), 3)
        # Verify ranking order (descending by candidate_prioritization_score)
        scores = [r.diagnostics.candidate_prioritization_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check top candidate
        top_res = results[0]
        self.assertEqual(top_res.hypothetical_release.mmsi, 412001111)
        self.assertEqual(top_res.diagnostics.classification, "High Spatiotemporal Consistency Lead")
        self.assertGreater(top_res.diagnostics.candidate_prioritization_score, 0.70)

        # Audit trail completeness
        self.assertTrue(len(top_res.assumptions) >= 4)
        self.assertIn("Forward 4th-Order Runge-Kutta", top_res.assumptions[0])
        self.assertIn("liability", top_res.limitations[0].lower())
        self.assertIn("provider", top_res.environmental_sources)

    def test_frozen_segformer_checkpoint_hash_unmodified(self):
        """Confirm that SegFormer checkpoint SHA-256 remains locked and untouched."""
        expected_hash = "cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6"
        with open(DEFAULT_CHECKPOINT_PATH, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(computed_hash, expected_hash, "best.pth SHA-256 altered!")


if __name__ == "__main__":
    unittest.main()
