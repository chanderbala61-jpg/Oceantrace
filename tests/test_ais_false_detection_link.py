"""
Unit and Integration Tests for OceanTrace AIS False-Detection Link Module
========================================================================
Tests cover:
  1. AIS schema loading & field alias mapping
  2. Timestamp parsing (ISO-8601, UTC offset handling)
  3. Coordinate validation & boundary checks (-90..90, -180..180)
  4. Spatial distance calculations (geodesic Haversine & segment projection)
  5. Temporal matching & lookback/lookforward windowing
  6. Candidate filtering within search radius
  7. Two-source AIS cross-check (MMSI overlap, domain independence)
  8. False-detection association & scientific status classification
  9. Missing AIS data handling (empty datasets, missing files)
  10. Missing coordinates handling (rejection without fabricating coords)
  11. Unreadable/invalid records handling (graceful skip and logging)
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.ais.schema import AISRecord, SpillEvent
from src.ais.loader import load_ais_csv
from src.ais.validator import validate_ais_record
from src.ais.spatial import haversine_distance_km, closest_distance_to_segment_km
from src.ais.temporal import filter_ais_by_time, get_spill_temporal_bounds
from src.ais.trajectory import VesselTrajectory, build_vessel_trajectories
from src.ais.features import extract_candidate_features
from src.ais.scoring import VesselScorer
from src.ais.false_detection_link import (
    FalseDetectionEvent,
    inspect_ais_dataset_quality,
    cross_check_two_ais_sources,
    analyze_event_correlation,
    STATUS_POTENTIAL_ASSOCIATION,
    STATUS_WEAK_ASSOCIATION,
    STATUS_INSUFFICIENT_DATA,
    STATUS_NO_MATCHING_VESSEL
)
from tests.fixtures.synthetic_ais import create_synthetic_ais_fixture


class TestAISFalseDetectionLink(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ds1_path = "outputs/investigations/00955_ais_traffic.csv"
        self.ds2_path = "see thsis/Ocean Trace 2/data/synthetic_ais_demo.csv"

        # Create temporary CSV with corrupt/missing rows for edge case testing
        self.corrupt_csv = os.path.join(self.tmp_dir, "corrupt_ais.csv")
        with open(self.corrupt_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,mmsi,latitude,longitude,sog,cog\n")
            f.write("2026-09-28T00:00:00Z,123456789,25.0,-90.0,10.0,45.0\n")  # Valid
            f.write("invalid_time,123456789,25.1,-90.1,10.0,45.0\n")             # Bad time
            f.write("2026-09-28T00:10:00Z,999,25.2,-90.2,10.0,45.0\n")           # Invalid MMSI
            f.write("2026-09-28T00:20:00Z,123456789,95.0,-90.3,10.0,45.0\n")     # Bad Lat (>90)
            f.write("2026-09-28T00:30:00Z,123456789,,,-90.4,10.0,45.0\n")        # Missing coords

    # 1. AIS Schema Loading & Quality Audit
    def test_ais_quality_audit_ds1(self):
        quality = inspect_ais_dataset_quality(self.ds1_path)
        self.assertEqual(quality["status"], "loaded")
        self.assertEqual(quality["unique_vessels"], 4)
        self.assertEqual(quality["missing_coords"], 0)
        self.assertGreater(quality["valid_records_loaded"], 10)

    def test_ais_quality_audit_ds2(self):
        quality = inspect_ais_dataset_quality(self.ds2_path)
        self.assertEqual(quality["status"], "loaded")
        self.assertEqual(quality["unique_vessels"], 4)
        self.assertEqual(quality["missing_coords"], 0)
        self.assertGreater(quality["valid_records_loaded"], 15)

    # 2. Timestamp Parsing
    def test_timestamp_parsing(self):
        records, _ = load_ais_csv(self.ds1_path)
        for r in records:
            self.assertIsInstance(r.timestamp, datetime)
            self.assertEqual(r.timestamp.year, 2019)

    # 3. Coordinate Validation
    def test_coordinate_validation(self):
        rec_valid = AISRecord(
            timestamp=datetime(2026, 9, 28, 0, 0, 0),
            mmsi=123456789,
            latitude=28.5,
            longitude=-90.5,
            sog=10.0,
            cog=90.0
        )
        is_val, errs = validate_ais_record(rec_valid)
        self.assertTrue(is_val)

        rec_invalid = AISRecord(
            timestamp=datetime(2026, 9, 28, 0, 0, 0),
            mmsi=123456789,
            latitude=91.0,
            longitude=-185.0,
            sog=10.0,
            cog=90.0
        )
        is_val2, errs2 = validate_ais_record(rec_invalid)
        self.assertFalse(is_val2)
        self.assertGreater(len(errs2), 0)

    # 4. Spatial Distance Calculation
    def test_haversine_distance(self):
        dist = haversine_distance_km(27.8420, -90.5099, 27.8500, -90.5000)
        self.assertAlmostEqual(dist, 1.32, delta=0.2)

    # 5. Temporal Matching
    def test_temporal_window_matching(self):
        event = FalseDetectionEvent(
            event_id="TEST_EVENT_001",
            scene_id="00955",
            timestamp=datetime(2019, 8, 7, 0, 25, 51),
            centroid_lat=27.8420,
            centroid_lon=-90.5099,
            fp_pixels=100
        )
        records, _ = load_ais_csv(self.ds1_path)
        spill = event.to_spill_event()
        t_start, t_end = get_spill_temporal_bounds(spill, lookback_hours=24.0, lookforward_hours=6.0)
        filt, stats = filter_ais_by_time(records, t_start, t_end)
        
        # 2019-08-07 records are included; 2019-08-03 records (3 days prior) are dropped
        mmsis = {r.mmsi for r in filt}
        self.assertIn(111222333, mmsis)
        self.assertNotIn(444555666, mmsis)  # 3 days earlier

    # 6. Candidate Filtering
    def test_candidate_filtering_and_scoring(self):
        event = FalseDetectionEvent(
            event_id="TEST_EVENT_001",
            scene_id="00955",
            timestamp=datetime(2019, 8, 7, 0, 25, 51),
            centroid_lat=27.8420,
            centroid_lon=-90.5099,
            estimated_origin_lat=27.8257,
            estimated_origin_lon=-90.5461,
            origin_uncertainty_km=8.0
        )
        records1, _ = load_ais_csv(self.ds1_path)
        records2, _ = load_ais_csv(self.ds2_path)
        cand_rows, summary = analyze_event_correlation(event, records1, records2)
        
        self.assertGreater(len(cand_rows), 0)
        self.assertEqual(summary["candidate_vessel_count"], 2)
        self.assertEqual(cand_rows[0]["mmsi"], 111222333)
        self.assertIn("Highly Compatible", cand_rows[0]["evidence_notes"])
        self.assertEqual(cand_rows[0]["candidate_status"], STATUS_POTENTIAL_ASSOCIATION)

    # 7. Two-Source AIS Cross-Check
    def test_two_source_cross_check(self):
        records1, _ = load_ais_csv(self.ds1_path)
        records2, _ = load_ais_csv(self.ds2_path)
        comparison = cross_check_two_ais_sources(records1, records2)
        
        # Both datasets are non-empty
        self.assertEqual(comparison["source_1_vessels"], 4)
        self.assertEqual(comparison["source_2_vessels"], 4)
        # Independent domains: zero overlap
        self.assertEqual(len(comparison["common_vessels"]), 0)
        self.assertFalse(comparison["geographic_overlap"])
        self.assertFalse(comparison["temporal_overlap"])

    # 8. False Detection Association with Zero Matching (Temporal Mismatch)
    def test_event_no_matching_vessel(self):
        # Event from 2017 (no AIS data available in 2019 or 2026)
        event_2017 = FalseDetectionEvent(
            event_id="FP_TEST_2017_00594",
            scene_id="00594",
            timestamp=datetime(2017, 4, 28, 0, 1, 42),
            centroid_lat=25.0,
            centroid_lon=-90.0,
            fp_pixels=48283,
            is_pure_fp=True
        )
        records1, _ = load_ais_csv(self.ds1_path)
        records2, _ = load_ais_csv(self.ds2_path)
        cand_rows, summary = analyze_event_correlation(event_2017, records1, records2)
        
        self.assertEqual(len(cand_rows), 0)
        self.assertEqual(summary["candidate_vessel_count"], 0)
        self.assertEqual(summary["correlation_status"], STATUS_NO_MATCHING_VESSEL)
        self.assertIn("No sufficient AIS evidence", summary["notes"])

    # 9. Missing AIS Data Handling
    def test_missing_ais_data(self):
        event = FalseDetectionEvent(
            event_id="TEST_EVENT_EMPTY",
            scene_id="00955",
            timestamp=datetime(2019, 8, 7, 0, 25, 51),
            centroid_lat=27.8420,
            centroid_lon=-90.5099
        )
        cand_rows, summary = analyze_event_correlation(event, [], [])
        self.assertEqual(len(cand_rows), 0)
        self.assertEqual(summary["correlation_status"], STATUS_NO_MATCHING_VESSEL)

    # 10. Missing Coordinates Handling
    def test_missing_coordinates_event(self):
        event_no_coords = FalseDetectionEvent(
            event_id="FP_NO_COORDS",
            scene_id="00317",
            timestamp=datetime(2019, 10, 26, 9, 52, 6),
            centroid_lat=None,  # Missing coordinates!
            centroid_lon=None,
            fp_pixels=1
        )
        records1, _ = load_ais_csv(self.ds1_path)
        cand_rows, summary = analyze_event_correlation(event_no_coords, records1, [])
        self.assertEqual(len(cand_rows), 0)
        self.assertEqual(summary["correlation_status"], STATUS_INSUFFICIENT_DATA)
        self.assertIn("Missing or unprojected geographic coordinates", summary["notes"])

    # 11. Unreadable/Corrupt Records Handling
    def test_corrupt_records_handling(self):
        records, summary = load_ais_csv(self.corrupt_csv)
        self.assertEqual(len(records), 1)  # Only 1 valid record
        self.assertGreater(summary["invalid_rows_skipped"], 0)


if __name__ == "__main__":
    unittest.main()
