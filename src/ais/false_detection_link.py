"""
OceanTrace: AIS & False-Detection Link Analysis Module
======================================================
Provides scientific, defensible linking between:
  False Detection / Detected Spill Event
          ↓
  Spatial + Temporal Context
          ↓
  AIS Vessel Tracks (Source 1 & Source 2)
          ↓
  Possible Candidate Vessels

CRITICAL SCIENTIFIC INTEGRITY RULES:
  - Does NOT assert legal responsibility or causality.
  - Candidate status uses only approved categories:
    'potential_spatiotemporal_association', 'weak_association',
    'insufficient_data', 'no_matching_vessel'.
  - Evaluates independent two-source AIS agreement without blind merging.
  - Distinguishes model false positives (segmentation error relative to GT)
    from possible real-world maritime activity.
"""

from __future__ import annotations

import os
import csv
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.ais.schema import AISRecord, SpillEvent, CandidateVesselResult
from src.ais.validator import validate_ais_record
from src.ais.loader import load_ais_csv
from src.ais.spatial import haversine_distance_km, closest_distance_to_segment_km
from src.ais.temporal import filter_ais_by_time, get_spill_temporal_bounds
from src.ais.trajectory import VesselTrajectory, build_vessel_trajectories
from src.ais.features import extract_candidate_features
from src.ais.scoring import VesselScorer, rank_candidate_vessels


from enum import Enum
from dataclasses import dataclass

# Approved candidate status categories
STATUS_POTENTIAL_ASSOCIATION = "potential_spatiotemporal_association"
STATUS_WEAK_ASSOCIATION = "weak_association"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_NO_MATCHING_VESSEL = "no_matching_vessel"


class FalseDetectionStatus(str, Enum):
    VALIDATED_OIL = "validated_oil"
    LOOK_ALIKE_AMBIGUOUS = "look_alike_ambiguous"
    AIS_COVERAGE_GAP = "ais_coverage_gap"
    FALSE_POSITIVE = "false_positive"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class SceneDiagnosticStatus:
    status: FalseDetectionStatus
    diagnostic_note: str
    is_false_positive: bool
    damping_contrast_db: Optional[float] = None
    ais_coverage_flag: str = "coverage_normal"


def evaluate_scene_false_detection_status(
    has_sar_detection: bool,
    damping_contrast_db: Optional[float] = None,
    candidate_vessels: Optional[List[Any]] = None,
    ais_coverage_available: bool = True,
    temporal_gap_flag: bool = False,
) -> SceneDiagnosticStatus:
    """
    Evaluates whether an observation represents a verified oil spill, a look-alike,
    or an AIS observation gap.

    SCIENTIFIC INTEGRITY RULE:
      Absence of AIS vessel tracks is an observation/coverage gap. It does NOT prove
      that a SAR detection is false.
    """
    vessels = candidate_vessels or []

    if not has_sar_detection:
        return SceneDiagnosticStatus(
            status=FalseDetectionStatus.FALSE_POSITIVE if damping_contrast_db is not None and damping_contrast_db > 0 else FalseDetectionStatus.INSUFFICIENT_DATA,
            diagnostic_note="No SAR oil detection confirmed.",
            is_false_positive=False,
            damping_contrast_db=damping_contrast_db,
            ais_coverage_flag="no_sar_detection"
        )

    # Check SAR contrast heuristic (< 3.5 dB is look-alike risk)
    if damping_contrast_db is not None and damping_contrast_db < 3.5:
        return SceneDiagnosticStatus(
            status=FalseDetectionStatus.LOOK_ALIKE_AMBIGUOUS,
            diagnostic_note=f"Low radiometric damping contrast ({damping_contrast_db:.2f} dB < 3.5 dB threshold). High probability of biogenic slick, grease ice, or low-wind look-alike.",
            is_false_positive=False,
            damping_contrast_db=damping_contrast_db,
            ais_coverage_flag="normal" if ais_coverage_available else "coverage_gap"
        )

    # If SAR detection exists and contrast is adequate or unmeasured:
    if not ais_coverage_available or temporal_gap_flag:
        return SceneDiagnosticStatus(
            status=FalseDetectionStatus.AIS_COVERAGE_GAP,
            diagnostic_note="SAR anomaly detected. AIS temporal or spatial coverage gap; absence of vessel transmissions does not refute spill presence.",
            is_false_positive=False,
            damping_contrast_db=damping_contrast_db,
            ais_coverage_flag="temporal_spatial_gap"
        )

    if not vessels:
        return SceneDiagnosticStatus(
            status=FalseDetectionStatus.VALIDATED_OIL,
            diagnostic_note="SAR anomaly detected with adequate contrast. No correlated AIS vessels in corridor (unmonitored, non-transmitting, or non-cooperative source).",
            is_false_positive=False,
            damping_contrast_db=damping_contrast_db,
            ais_coverage_flag="zero_candidates"
        )

    return SceneDiagnosticStatus(
        status=FalseDetectionStatus.VALIDATED_OIL,
        diagnostic_note=f"SAR oil detection correlated with {len(vessels)} candidate vessels in corridor.",
        is_false_positive=False,
        damping_contrast_db=damping_contrast_db,
        ais_coverage_flag="candidates_present"
    )


class FalseDetectionEvent:
    """Represents a false detection or detected spill event for AIS correlation."""
    def __init__(
        self,
        event_id: str,
        scene_id: str,
        timestamp: Optional[datetime],
        centroid_lat: Optional[float],
        centroid_lon: Optional[float],
        fp_pixels: int = 0,
        fp_area_km2: Optional[float] = None,
        is_pure_fp: bool = False,
        estimated_origin_lat: Optional[float] = None,
        estimated_origin_lon: Optional[float] = None,
        origin_time_start: Optional[datetime] = None,
        origin_time_end: Optional[datetime] = None,
        origin_uncertainty_km: float = 10.0,
        metadata_notes: str = ""
    ):
        self.event_id = event_id
        self.scene_id = scene_id
        self.timestamp = timestamp
        self.centroid_lat = centroid_lat
        self.centroid_lon = centroid_lon
        self.fp_pixels = fp_pixels
        self.fp_area_km2 = fp_area_km2
        self.is_pure_fp = is_pure_fp
        self.estimated_origin_lat = estimated_origin_lat or centroid_lat
        self.estimated_origin_lon = estimated_origin_lon or centroid_lon
        self.origin_time_start = origin_time_start
        self.origin_time_end = origin_time_end
        self.origin_uncertainty_km = origin_uncertainty_km
        self.metadata_notes = metadata_notes

    @property
    def has_valid_coordinates(self) -> bool:
        if self.centroid_lat is None or self.centroid_lon is None:
            return False
        return -90.0 <= self.centroid_lat <= 90.0 and -180.0 <= self.centroid_lon <= 180.0

    @property
    def has_valid_timestamp(self) -> bool:
        return self.timestamp is not None

    def to_spill_event(self) -> Optional[SpillEvent]:
        if not self.has_valid_coordinates or not self.has_valid_timestamp:
            return None
        return SpillEvent(
            spill_id=self.event_id,
            observation_time=self.timestamp,
            centroid_lat=self.centroid_lat,
            centroid_lon=self.centroid_lon,
            area_km2=self.fp_area_km2 or 0.0,
            estimated_origin_lat=self.estimated_origin_lat,
            estimated_origin_lon=self.estimated_origin_lon,
            origin_time_start=self.origin_time_start,
            origin_time_end=self.origin_time_end,
            origin_uncertainty_km=self.origin_uncertainty_km
        )


def inspect_ais_dataset_quality(csv_path: str) -> Dict[str, Any]:
    """
    Audits AIS dataset quality in strict accordance with Step 2:
    - row count, vessel count
    - timestamp parsing and timezone detection
    - coordinate validity and bounds
    - missing fields, duplicates, invalid values
    """
    if not os.path.exists(csv_path):
        return {
            "status": "missing_file",
            "path": csv_path,
            "error": "File does not exist"
        }

    records, summary = load_ais_csv(csv_path)
    file_size = os.path.getsize(csv_path)

    # Re-read raw rows for field audit
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
        raw_cols = reader.fieldnames or []

    unique_mmsis = set()
    invalid_coords = 0
    missing_coords = 0
    missing_timestamps = 0
    duplicate_rows = 0
    seen_keys = set()
    lats = []
    lons = []
    timestamps = []
    tz_aware_count = 0

    for r in raw_rows:
        # Check duplicate by (mmsi, timestamp)
        mmsi_raw = r.get("mmsi") or r.get("vessel_mmsi", "")
        ts_raw = r.get("timestamp") or r.get("basedatetime", "")
        key = (mmsi_raw, ts_raw)
        if key in seen_keys:
            duplicate_rows += 1
        seen_keys.add(key)

        # Coordinate checks
        lat_raw = r.get("latitude") or r.get("lat")
        lon_raw = r.get("longitude") or r.get("lon") or r.get("long")
        if not lat_raw or not lon_raw:
            missing_coords += 1
        else:
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    lats.append(lat)
                    lons.append(lon)
                else:
                    invalid_coords += 1
            except ValueError:
                invalid_coords += 1

        # Timestamp checks
        if not ts_raw:
            missing_timestamps += 1
        else:
            if "Z" in ts_raw or "+" in ts_raw:
                tz_aware_count += 1

    for rec in records:
        unique_mmsis.add(rec.mmsi)
        timestamps.append(rec.timestamp)

    return {
        "status": "loaded",
        "path": csv_path,
        "file_size_bytes": file_size,
        "columns": raw_cols,
        "total_rows": len(raw_rows),
        "valid_records_loaded": len(records),
        "invalid_rows_skipped": summary.get("invalid_rows_skipped", 0),
        "unique_vessels": len(unique_mmsis),
        "unique_mmsis": sorted(list(unique_mmsis)),
        "missing_coords": missing_coords,
        "invalid_coords": invalid_coords,
        "missing_timestamps": missing_timestamps,
        "duplicate_records": duplicate_rows,
        "lat_range": (min(lats), max(lats)) if lats else None,
        "lon_range": (min(lons), max(lons)) if lons else None,
        "temporal_range": (min(timestamps), max(timestamps)) if timestamps else None,
        "tz_aware_ratio": tz_aware_count / len(raw_rows) if raw_rows else 0.0
    }


def cross_check_two_ais_sources(
    ds1_records: List[AISRecord],
    ds2_records: List[AISRecord]
) -> Dict[str, Any]:
    """
    Step 7: Compares two AIS datasets without blind merging.
    Inspects:
      - MMSI overlap
      - Geographic overlap
      - Temporal overlap
      - Concordance of positions for any common vessels
    """
    mmsi1 = set(r.mmsi for r in ds1_records)
    mmsi2 = set(r.mmsi for r in ds2_records)
    common_mmsi = mmsi1.intersection(mmsi2)

    # Geographic bounding boxes
    b1_lat = (min(r.latitude for r in ds1_records), max(r.latitude for r in ds1_records)) if ds1_records else None
    b1_lon = (min(r.longitude for r in ds1_records), max(r.longitude for r in ds1_records)) if ds1_records else None
    b2_lat = (min(r.latitude for r in ds2_records), max(r.latitude for r in ds2_records)) if ds2_records else None
    b2_lon = (min(r.longitude for r in ds2_records), max(r.longitude for r in ds2_records)) if ds2_records else None

    # Temporal bounds
    t1 = (min(r.timestamp for r in ds1_records), max(r.timestamp for r in ds1_records)) if ds1_records else None
    t2 = (min(r.timestamp for r in ds2_records), max(r.timestamp for r in ds2_records)) if ds2_records else None

    geo_overlap = False
    if b1_lat and b2_lat and b1_lon and b2_lon:
        lat_overlap = not (b1_lat[1] < b2_lat[0] or b2_lat[1] < b1_lat[0])
        lon_overlap = not (b1_lon[1] < b2_lon[0] or b2_lon[1] < b1_lon[0])
        geo_overlap = lat_overlap and lon_overlap

    temp_overlap = False
    if t1 and t2:
        temp_overlap = not (t1[1] < t2[0] or t2[1] < t1[0])

    return {
        "source_1_record_count": len(ds1_records),
        "source_2_record_count": len(ds2_records),
        "source_1_vessels": len(mmsi1),
        "source_2_vessels": len(mmsi2),
        "common_vessels": sorted(list(common_mmsi)),
        "geographic_overlap": geo_overlap,
        "temporal_overlap": temp_overlap,
        "independent_agreement": "independent_domains_no_overlap" if not common_mmsi and not geo_overlap else "evaluated"
    }


def analyze_event_correlation(
    event: FalseDetectionEvent,
    ds1_records: List[AISRecord],
    ds2_records: List[AISRecord],
    lookback_hours: float = 24.0,
    lookforward_hours: float = 6.0,
    max_search_radius_km: float = 50.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Correlates a false-detection or detected spill event with both AIS datasets.
    Returns:
      (candidate_correlations_rows, event_summary_dict)
    """
    candidate_rows = []

    if not event.has_valid_coordinates:
        summary = {
            "event_id": event.event_id,
            "scene_id": event.scene_id,
            "false_positive_area": event.fp_area_km2 or f"{event.fp_pixels} px",
            "event_timestamp": event.timestamp.isoformat() if event.timestamp else "N/A",
            "candidate_vessel_count": 0,
            "nearest_vessel_mmsi": "N/A",
            "nearest_vessel_distance_km": "N/A",
            "nearest_vessel_time_difference_minutes": "N/A",
            "trajectory_evidence": "none",
            "ais_source_agreement": "no_coordinates",
            "correlation_status": STATUS_INSUFFICIENT_DATA,
            "notes": "Missing or unprojected geographic coordinates. Pixel coordinates cannot be correlated to AIS."
        }
        return candidate_rows, summary

    if not event.has_valid_timestamp:
        summary = {
            "event_id": event.event_id,
            "scene_id": event.scene_id,
            "false_positive_area": event.fp_area_km2 or f"{event.fp_pixels} px",
            "event_timestamp": "N/A",
            "candidate_vessel_count": 0,
            "nearest_vessel_mmsi": "N/A",
            "nearest_vessel_distance_km": "N/A",
            "nearest_vessel_time_difference_minutes": "N/A",
            "trajectory_evidence": "none",
            "ais_source_agreement": "no_timestamp",
            "correlation_status": STATUS_INSUFFICIENT_DATA,
            "notes": "Missing observation timestamp. Temporal window cannot be established."
        }
        return candidate_rows, summary

    def _normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    def _norm_record(rec: AISRecord) -> AISRecord:
        if rec.timestamp.tzinfo is not None:
            return AISRecord(
                timestamp=_normalize_dt(rec.timestamp),
                mmsi=rec.mmsi,
                latitude=rec.latitude,
                longitude=rec.longitude,
                sog=rec.sog,
                cog=rec.cog,
                heading=rec.heading,
                nav_status=rec.nav_status,
                ship_name=rec.ship_name,
                ship_type=rec.ship_type,
                imo=rec.imo,
                length=rec.length,
                width=rec.width,
                draught=rec.draught
            )
        return rec

    norm_event_ts = _normalize_dt(event.timestamp)
    norm_start = _normalize_dt(event.origin_time_start)
    norm_end = _normalize_dt(event.origin_time_end)

    spill = SpillEvent(
        spill_id=event.event_id,
        observation_time=norm_event_ts,
        centroid_lat=event.centroid_lat,
        centroid_lon=event.centroid_lon,
        area_km2=event.fp_area_km2 or 0.0,
        estimated_origin_lat=event.estimated_origin_lat,
        estimated_origin_lon=event.estimated_origin_lon,
        origin_time_start=norm_start,
        origin_time_end=norm_end,
        origin_uncertainty_km=event.origin_uncertainty_km
    )
    t_start, t_end = get_spill_temporal_bounds(spill, lookback_hours, lookforward_hours)

    norm_ds1 = [_norm_record(r) for r in ds1_records]
    norm_ds2 = [_norm_record(r) for r in ds2_records]

    # Filter DS1 and DS2 by time window
    filt_ds1, stats1 = filter_ais_by_time(norm_ds1, t_start, t_end)
    filt_ds2, stats2 = filter_ais_by_time(norm_ds2, t_start, t_end)

    trajs1 = build_vessel_trajectories(filt_ds1)
    trajs2 = build_vessel_trajectories(filt_ds2)

    scorer = VesselScorer(max_search_radius_km=max_search_radius_km, max_time_window_hours=lookback_hours)

    all_vessels = []
    # Process DS1 vessels
    for mmsi, traj in trajs1.items():
        feat = extract_candidate_features(traj, spill)
        min_dist = feat["min_distance_km"]
        if min_dist <= max_search_radius_km:
            scored = scorer.score_candidate(feat, spill)
            all_vessels.append(("Dataset_1", traj, feat, scored))

    # Process DS2 vessels
    for mmsi, traj in trajs2.items():
        feat = extract_candidate_features(traj, spill)
        min_dist = feat["min_distance_km"]
        if min_dist <= max_search_radius_km:
            scored = scorer.score_candidate(feat, spill)
            all_vessels.append(("Dataset_2", traj, feat, scored))

    # Sort all candidates by score descending
    all_vessels.sort(key=lambda x: x[3].overall_score, reverse=True)

    if not all_vessels:
        summary = {
            "event_id": event.event_id,
            "scene_id": event.scene_id,
            "false_positive_area": event.fp_area_km2 or f"{event.fp_pixels} px",
            "event_timestamp": event.timestamp.isoformat(),
            "candidate_vessel_count": 0,
            "nearest_vessel_mmsi": "N/A",
            "nearest_vessel_distance_km": "N/A",
            "nearest_vessel_time_difference_minutes": "N/A",
            "trajectory_evidence": "no_vessels_in_window",
            "ais_source_agreement": "no_data_both_sources" if not filt_ds1 and not filt_ds2 else "no_vessels_within_radius",
            "correlation_status": STATUS_NO_MATCHING_VESSEL,
            "notes": (
                "No sufficient AIS evidence was available for this event within configured search window. "
                "AIS data absence reflects a temporal/geographic coverage gap or non-transmitting traffic; "
                "it does NOT constitute proof that the SAR detection is false."
            )
        }
        return candidate_rows, summary

    # Build candidate correlation rows
    nearest_dist = float("inf")
    nearest_mmsi = None
    nearest_time_diff_min = None
    best_traj_relation = "insufficient_coverage"
    best_status = STATUS_WEAK_ASSOCIATION

    for src_name, traj, feat, scored in all_vessels:
        mmsi = traj.mmsi
        cpa_dist = feat["min_distance_km"]
        cpa_time = feat["closest_approach_time"]
        time_diff_min = (_normalize_dt(cpa_time) - norm_event_ts).total_seconds() / 60.0 if cpa_time else 0.0

        if cpa_dist < nearest_dist:
            nearest_dist = cpa_dist
            nearest_mmsi = mmsi
            nearest_time_diff_min = round(time_diff_min, 1)

        # Trajectory relationship classification
        if cpa_dist <= event.origin_uncertainty_km:
            if feat["mean_sog"] >= 5.0:
                traj_relation = "passes_through_or_near_detection_region"
            else:
                traj_relation = "remains_nearby_slow_movement"
        elif cpa_dist <= max_search_radius_km:
            traj_relation = "approaches_and_moves_away"
        else:
            traj_relation = "insufficient_coverage"

        # Candidate status
        if scored.overall_score >= 0.75:
            cand_status = STATUS_POTENTIAL_ASSOCIATION
        elif scored.overall_score >= 0.45:
            cand_status = STATUS_WEAK_ASSOCIATION
        else:
            cand_status = STATUS_WEAK_ASSOCIATION

        # Check two-source support for this MMSI
        in_ds1 = any(r.mmsi == mmsi for r in ds1_records)
        in_ds2 = any(r.mmsi == mmsi for r in ds2_records)

        evidence_notes = (
            f"Compatibility score={scored.overall_score:.2f} ({scored.classification}). "
            f"CPA={cpa_dist:.2f} km, Time offset={time_diff_min:.1f} min, SOG={feat['mean_sog']:.1f} kn. "
            f"Source={src_name}."
        )

        candidate_rows.append({
            "event_id": event.event_id,
            "scene_id": event.scene_id,
            "detection_timestamp": event.timestamp.isoformat(),
            "detection_lat": round(event.centroid_lat, 6),
            "detection_lon": round(event.centroid_lon, 6),
            "mmsi": mmsi,
            "vessel_name_if_available": traj.ship_name or "UNKNOWN",
            "ais_source": src_name,
            "ais_timestamp": cpa_time.isoformat() if cpa_time else "N/A",
            "ais_lat": round(feat.get("closest_lat", event.centroid_lat), 6),
            "ais_lon": round(feat.get("closest_lon", event.centroid_lon), 6),
            "distance_km": round(cpa_dist, 3),
            "time_difference_minutes": round(time_diff_min, 1),
            "trajectory_relation": traj_relation,
            "dataset_1_support": "present" if in_ds1 else "absent",
            "dataset_2_support": "present" if in_ds2 else "absent",
            "candidate_status": cand_status,
            "evidence_notes": evidence_notes
        })

    # Summary
    top_cand = all_vessels[0]
    top_score = top_cand[3].overall_score
    if top_score >= 0.75:
        overall_corr_status = STATUS_POTENTIAL_ASSOCIATION
    elif top_score >= 0.45:
        overall_corr_status = STATUS_WEAK_ASSOCIATION
    else:
        overall_corr_status = STATUS_NO_MATCHING_VESSEL

    ds1_vessels_in_event = sum(1 for v in all_vessels if v[0] == "Dataset_1")
    ds2_vessels_in_event = sum(1 for v in all_vessels if v[0] == "Dataset_2")
    if ds1_vessels_in_event > 0 and ds2_vessels_in_event > 0:
        agreement_desc = "both_sources_contain_candidates"
    elif ds1_vessels_in_event > 0:
        agreement_desc = "dataset_1_only"
    elif ds2_vessels_in_event > 0:
        agreement_desc = "dataset_2_only"
    else:
        agreement_desc = "neither_source"

    summary = {
        "event_id": event.event_id,
        "scene_id": event.scene_id,
        "false_positive_area": event.fp_area_km2 or f"{event.fp_pixels} px",
        "event_timestamp": event.timestamp.isoformat(),
        "candidate_vessel_count": len(all_vessels),
        "nearest_vessel_mmsi": nearest_mmsi,
        "nearest_vessel_distance_km": round(nearest_dist, 3),
        "nearest_vessel_time_difference_minutes": nearest_time_diff_min,
        "trajectory_evidence": all_vessels[0][2].get("trajectory_relation", "spatiotemporal_cpa_evaluated"),
        "ais_source_agreement": agreement_desc,
        "correlation_status": overall_corr_status,
        "notes": (
            f"{len(all_vessels)} candidate vessels evaluated. "
            f"Nearest vessel MMSI {nearest_mmsi} passed within {nearest_dist:.2f} km. "
            f"Candidate association reflects spatial-temporal transit compatibility only; "
            f"does not establish legal or physical causation."
        )
    }

    return candidate_rows, summary
