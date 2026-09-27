"""
OceanTrace AIS Correlation Module Package Entry Point
-----------------------------------------------------
Exposes core data structures and pipeline orchestrators.
"""

from src.ais.schema import AISRecord, SpillEvent, CandidateVesselResult
from src.ais.validator import validate_ais_record, AISValidationError
from src.ais.loader import load_ais_csv
from src.ais.spatial import haversine_distance_km, closest_distance_to_segment_km
from src.ais.temporal import filter_ais_by_time, get_spill_temporal_bounds
from src.ais.trajectory import VesselTrajectory, build_vessel_trajectories
from src.ais.features import extract_candidate_features
from src.ais.scoring import VesselScorer, rank_candidate_vessels
from src.ais.hindcasting_interface import HindcastingModelInterface, DirectObservationBaseline
from src.ais.visualization import export_correlation_to_geojson, plot_ais_correlation_map
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


def run_ais_correlation_pipeline(
    spill: SpillEvent,
    ais_csv_path: str,
    lookback_hours: float = 24.0,
    lookforward_hours: float = 6.0,
    search_radius_km: float = 50.0
):
    """
    End-to-end AIS correlation orchestrator.
    
    Loads AIS records, filters by temporal and spatial windows, builds vessel
    trajectories, extracts evidence features, and scores candidate vessels.
    """
    # 1. Load AIS records
    records, load_summary = load_ais_csv(ais_csv_path)

    # 2. Temporal filter
    t_start, t_end = get_spill_temporal_bounds(spill, lookback_hours, lookforward_hours)
    time_filtered, time_stats = filter_ais_by_time(records, t_start, t_end)

    # 3. Build trajectories
    all_trajectories = build_vessel_trajectories(time_filtered)

    # 4. Spatial filter & feature extraction
    target_lat = spill.effective_origin_lat
    target_lon = spill.effective_origin_lon
    candidate_features = []
    filtered_trajectories = {}

    for mmsi, traj in all_trajectories.items():
        feat = extract_candidate_features(traj, spill)
        # Check if closest approach is within search radius
        if feat["min_distance_km"] <= search_radius_km:
            candidate_features.append(feat)
            filtered_trajectories[mmsi] = traj

    # 5. Score and rank candidates
    scorer = VesselScorer(max_search_radius_km=search_radius_km)
    ranked_candidates = rank_candidate_vessels(candidate_features, spill, scorer)

    return {
        "spill": spill,
        "load_summary": load_summary,
        "time_stats": time_stats,
        "total_vessels_in_time_window": len(all_trajectories),
        "candidates_within_search_radius": len(candidate_features),
        "ranked_candidates": ranked_candidates,
        "candidate_trajectories": filtered_trajectories
    }
