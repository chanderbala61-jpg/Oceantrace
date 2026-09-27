"""
Comprehensive Unit Test Suite for OceanTrace AIS Correlation Module
-------------------------------------------------------------------
Tests:
  1. AIS schema validation
  2. Invalid coordinates & physical limits
  3. Timestamp parsing formats
  4. Temporal filtering logic
  5. Spatial distance calculation (Haversine & segment projection)
  6. Trajectory reconstruction & closest point of approach
  7. Candidate filtering & spatial radius thresholding
  8. Evidence feature extraction
  9. Explainable scoring & ranking order
  10. Non-accusatory classification policy
  11. Empty AIS dataset handling
  12. Missing optional fields tolerance
  13. Multiple vessels ranking
  14. GeoJSON export integrity
  15. Hindcasting interface fallback
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from src.ais.schema import AISRecord, SpillEvent, CandidateVesselResult
from src.ais.validator import validate_ais_record
from src.ais.loader import load_ais_csv
from src.ais.spatial import haversine_distance_km, closest_distance_to_segment_km
from src.ais.temporal import filter_ais_by_time, get_spill_temporal_bounds
from src.ais.trajectory import VesselTrajectory, build_vessel_trajectories
from src.ais.features import extract_candidate_features
from src.ais.scoring import VesselScorer, rank_candidate_vessels
from src.ais.hindcasting_interface import DirectObservationBaseline
from src.ais.visualization import export_correlation_to_geojson
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture


class TestAISCorrelationModule(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.fixture_csv = os.path.join(self.tmp_dir, "synthetic_ais.csv")
        create_synthetic_ais_fixture(self.fixture_csv)

        self.spill = SpillEvent(
            spill_id="SPILL_2026_001",
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            centroid_lat=28.50,
            centroid_lon=-88.50,
            area_km2=14.2,
            estimated_origin_lat=28.50,
            estimated_origin_lon=-88.50,
            origin_time_start=datetime(2026, 9, 4, 10, 0, 0),
            origin_time_end=datetime(2026, 9, 4, 14, 0, 0),
            origin_uncertainty_km=10.0
        )

    # 1. Schema Validation
    def test_valid_ais_record(self):
        rec = AISRecord(
            timestamp=datetime(2026, 9, 4, 12, 0, 0),
            mmsi=123456789,
            latitude=25.0,
            longitude=-80.0,
            sog=12.5,
            cog=90.0
        )
        is_valid, errs = validate_ais_record(rec)
        self.assertTrue(is_valid)
        self.assertEqual(len(errs), 0)

    # 2. Invalid coordinates
    def test_invalid_coordinates(self):
        rec = AISRecord(
            timestamp=datetime(2026, 9, 4, 12, 0, 0),
            mmsi=123456789,
            latitude=95.0,  # Invalid
            longitude=-190.0,  # Invalid
            sog=10.0,
            cog=0.0
        )
        is_valid, errs = validate_ais_record(rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("Latitude" in e for e in errs))
        self.assertTrue(any("Longitude" in e for e in errs))

    # 3. Invalid MMSI & SOG
    def test_invalid_mmsi_and_sog(self):
        rec = AISRecord(
            timestamp=datetime(2026, 9, 4, 12, 0, 0),
            mmsi=123,  # Invalid
            latitude=0.0,
            longitude=0.0,
            sog=-5.0,  # Invalid
            cog=0.0
        )
        is_valid, errs = validate_ais_record(rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("MMSI" in e for e in errs))
        self.assertTrue(any("SOG" in e for e in errs))

    # 4. Spatial Math: Haversine distance
    def test_haversine_distance(self):
        # Distance between (0, 0) and (0, 1) deg lon at equator is ~111.19 km
        d = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(d, 111.19, delta=0.5)

        # Distance to self is 0
        d0 = haversine_distance_km(28.5, -88.5, 28.5, -88.5)
        self.assertEqual(d0, 0.0)

    # 5. Spatial Segment Distance
    def test_segment_closest_distance(self):
        # Point P at (0, 0), segment from (-1, -1) to (1, 1) crosses (0,0)
        dist, c_lat, c_lon = closest_distance_to_segment_km(
            0.0, 0.0,
            -0.1, -0.1,
            0.1, 0.1
        )
        self.assertAlmostEqual(dist, 0.0, delta=0.1)

    # 6. CSV Loading
    def test_csv_loader(self):
        records, summary = load_ais_csv(self.fixture_csv)
        self.assertGreater(len(records), 0)
        self.assertEqual(summary["invalid_rows_skipped"], 0)

    # 7. Temporal Filtering
    def test_temporal_filter(self):
        records, _ = load_ais_csv(self.fixture_csv)
        t_start, t_end = get_spill_temporal_bounds(self.spill, lookback_hours=12.0, lookforward_hours=6.0)
        filtered, stats = filter_ais_by_time(records, t_start, t_end)
        
        # MMSI 444555666 was 3 days earlier and must be filtered out
        mmsis = {r.mmsi for r in filtered}
        self.assertNotIn(444555666, mmsis)
        self.assertIn(111222333, mmsis)

    # 8. Trajectory Closest Approach
    def test_trajectory_closest_approach(self):
        records, _ = load_ais_csv(self.fixture_csv)
        trajectories = build_vessel_trajectories(records)
        traj_alpha = trajectories[111222333]
        
        approach = traj_alpha.compute_closest_approach(28.50, -88.50)
        self.assertAlmostEqual(approach["min_distance_km"], 0.0, delta=1.0)
        self.assertIsNotNone(approach["closest_approach_time"])

    # 9. Feature Extraction
    def test_feature_extraction(self):
        records, _ = load_ais_csv(self.fixture_csv)
        trajectories = build_vessel_trajectories(records)
        feat = extract_candidate_features(trajectories[111222333], self.spill)
        
        self.assertEqual(feat["mmsi"], 111222333)
        self.assertTrue(feat["within_uncertainty_zone"])
        self.assertLess(feat["time_difference_hours"], 1.0)

    # 10. Scoring & Ranking Logic
    def test_scoring_and_ranking(self):
        records, _ = load_ais_csv(self.fixture_csv)
        # Apply time filter
        t_start, t_end = get_spill_temporal_bounds(self.spill, lookback_hours=12.0, lookforward_hours=6.0)
        filtered, _ = filter_ais_by_time(records, t_start, t_end)
        trajectories = build_vessel_trajectories(filtered)
        
        features = [extract_candidate_features(t, self.spill) for t in trajectories.values()]
        # Filter within 50km
        candidates = [f for f in features if f["min_distance_km"] <= 50.0]
        
        scorer = VesselScorer()
        ranked = rank_candidate_vessels(candidates, self.spill, scorer)
        
        self.assertGreaterEqual(len(ranked), 2)
        # Rank 1 must be MMSI 111222333 (high compatibility transit)
        self.assertEqual(ranked[0].mmsi, 111222333)
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[0].classification, "Highly Compatible Candidate")
        self.assertGreaterEqual(ranked[0].overall_score, 0.85)

    # 11. Strict Non-Accusatory Policy Check
    def test_non_accusatory_language(self):
        records, _ = load_ais_csv(self.fixture_csv)
        trajectories = build_vessel_trajectories(records)
        feat = extract_candidate_features(trajectories[111222333], self.spill)
        scorer = VesselScorer()
        res = scorer.score_candidate(feat, self.spill)
        
        self.assertNotIn("guilty", res.explanation.lower())
        self.assertNotIn("caused the spill", res.explanation.lower())
        self.assertIn("compatible", res.classification.lower())

    # 12. Empty Dataset Handling
    def test_empty_ais_handling(self):
        empty_csv = os.path.join(self.tmp_dir, "empty.csv")
        with open(empty_csv, "w") as f:
            f.write("timestamp,mmsi,latitude,longitude,sog,cog\n")
            
        records, summary = load_ais_csv(empty_csv)
        self.assertEqual(len(records), 0)
        self.assertEqual(summary["total_rows_read"], 0)

    # 13. Missing Optional Fields Tolerance
    def test_missing_optional_fields(self):
        csv_p = os.path.join(self.tmp_dir, "minimal.csv")
        with open(csv_p, "w") as f:
            f.write("timestamp,mmsi,latitude,longitude,sog,cog\n")
            f.write("2026-09-04T12:00:00Z,999888777,28.5,-88.5,10.0,45.0\n")
            
        records, summary = load_ais_csv(csv_p)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].ship_name)
        self.assertIsNone(records[0].heading)

    # 14. GeoJSON Export Verification
    def test_geojson_export(self):
        records, _ = load_ais_csv(self.fixture_csv)
        trajectories = build_vessel_trajectories(records)
        features = [extract_candidate_features(t, self.spill) for t in trajectories.values()]
        ranked = rank_candidate_vessels(features, self.spill)
        
        out_geojson = os.path.join(self.tmp_dir, "correlation.geojson")
        export_correlation_to_geojson(self.spill, trajectories, ranked, out_geojson)
        
        self.assertTrue(os.path.exists(out_geojson))
        self.assertGreater(os.path.getsize(out_geojson), 100)

    # 15. Hindcasting Baseline Fallback
    def test_hindcasting_fallback(self):
        spill_unhindcasted = SpillEvent(
            spill_id="SPILL_RAW",
            observation_time=datetime(2026, 9, 4, 12, 0, 0),
            centroid_lat=28.50,
            centroid_lon=-88.50,
            area_km2=5.0
        )
        self.assertIsNone(spill_unhindcasted.estimated_origin_lat)
        
        engine = DirectObservationBaseline()
        spill_updated = engine.estimate_origin(spill_unhindcasted, drift_hours=12.0)
        
        self.assertEqual(spill_updated.effective_origin_lat, 28.50)
        self.assertEqual(spill_updated.effective_origin_lon, -88.50)
        self.assertIsNotNone(spill_updated.origin_time_start)


if __name__ == "__main__":
    unittest.main()
