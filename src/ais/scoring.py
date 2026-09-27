"""
Explainable Multi-Factor Scoring and Candidate Ranking
------------------------------------------------------
Calculates an explainable compatibility score for candidate vessels based on:
  - Spatial proximity
  - Temporal alignment
  - Trajectory / uncertainty intersection
  - Kinematic consistency

CRITICAL POLICY:
  - Never outputs 'This vessel caused the spill.'
  - Classifies candidates as 'Highly Compatible', 'Moderately Compatible', or 'Low Compatibility'.
"""

from typing import Dict, Any, List, Optional
from src.ais.schema import CandidateVesselResult, SpillEvent


# Default scoring factor weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "spatial": 0.35,      # Proximity to estimated spill origin
    "temporal": 0.30,     # Closeness to estimated release window
    "trajectory": 0.20,   # Passage through the origin uncertainty radius
    "kinematics": 0.15    # Normal transit vs abnormal behavior
}


class VesselScorer:
    """
    Evaluates extracted evidence features into normalized [0.0, 1.0] factor scores
    and computes the weighted overall compatibility score.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_search_radius_km: float = 50.0,
        max_time_window_hours: float = 24.0
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.max_radius = max_search_radius_km
        self.max_hours = max_time_window_hours

        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_candidate(
        self,
        features: Dict[str, Any],
        spill: SpillEvent,
        rank: int = 1
    ) -> CandidateVesselResult:
        """
        Computes component evidence scores and overall compatibility score.
        """
        min_dist = features["min_distance_km"]
        time_diff = features["time_difference_hours"]
        within_zone = features["within_uncertainty_zone"]
        mean_sog = features["mean_sog"]
        uncertainty_km = spill.origin_uncertainty_km

        # 1. Spatial Proximity Score [0, 1]:
        # Full score (1.0) if directly inside uncertainty radius; linear decay to 0 at max_radius.
        if min_dist <= uncertainty_km:
            s_spatial = 1.0
        else:
            remaining_dist = min_dist - uncertainty_km
            decay_scale = max(1.0, self.max_radius - uncertainty_km)
            s_spatial = max(0.0, 1.0 - (remaining_dist / decay_scale))

        # 2. Temporal Proximity Score [0, 1]:
        # Decay linearly with time difference from the release window center.
        s_temporal = max(0.0, 1.0 - (time_diff / self.max_hours))

        # 3. Trajectory / Direct Intersect Score [0, 1]:
        # 1.0 if trajectory enters the origin uncertainty buffer, otherwise proportional to closest approach.
        if within_zone:
            s_trajectory = 1.0
        else:
            s_trajectory = max(0.0, 1.0 - (min_dist / (2.0 * max(1.0, uncertainty_km))))

        # 4. Kinematic Consistency Score [0, 1]:
        # Typical commercial transit (6 to 25 knots) scores high; very slow drifting / loitering (<2 kn) or stopped flags caution.
        if 6.0 <= mean_sog <= 25.0:
            s_kinematics = 1.0
        elif 2.0 <= mean_sog < 6.0:
            s_kinematics = 0.7  # Slow moving
        elif mean_sog < 2.0:
            s_kinematics = 0.4  # Loitering / stationary
        else:
            s_kinematics = 0.6  # High-speed vessel

        # Overall weighted score
        overall = (
            self.weights["spatial"] * s_spatial +
            self.weights["temporal"] * s_temporal +
            self.weights["trajectory"] * s_trajectory +
            self.weights["kinematics"] * s_kinematics
        )
        overall = round(max(0.0, min(1.0, overall)), 4)

        # Classification label
        if overall >= 0.75:
            classification = "Highly Compatible Candidate"
            summary_desc = (
                f"Vessel passed within {min_dist} km of the spill origin "
                f"approximately {time_diff} hours from the estimated release window, "
                f"demonstrating strong spatial and temporal correlation."
            )
        elif overall >= 0.45:
            classification = "Moderately Compatible Candidate"
            summary_desc = (
                f"Vessel was within {min_dist} km of the origin region "
                f"with moderate temporal proximity ({time_diff} hr difference)."
            )
        else:
            classification = "Low Compatibility Candidate"
            summary_desc = (
                f"Vessel trajectory showed marginal correlation "
                f"(distance: {min_dist} km, time difference: {time_diff} hr)."
            )

        evidence_scores = {
            "spatial_proximity": round(s_spatial, 4),
            "temporal_alignment": round(s_temporal, 4),
            "trajectory_overlap": round(s_trajectory, 4),
            "kinematic_consistency": round(s_kinematics, 4)
        }

        return CandidateVesselResult(
            mmsi=features["mmsi"],
            rank=rank,
            overall_score=overall,
            classification=classification,
            evidence_scores=evidence_scores,
            measurements=features,
            explanation=summary_desc,
            audit_trail={
                "weights_used": self.weights,
                "uncertainty_radius_km": uncertainty_km,
                "search_radius_km": self.max_radius
            }
        )


def rank_candidate_vessels(
    candidate_features: List[Dict[str, Any]],
    spill: SpillEvent,
    scorer: Optional[VesselScorer] = None
) -> List[CandidateVesselResult]:
    """
    Scores and ranks all candidate vessels in descending order of compatibility score.
    """
    if scorer is None:
        scorer = VesselScorer()

    results = []
    for feat in candidate_features:
        res = scorer.score_candidate(feat, spill, rank=0)
        results.append(res)

    # Sort descending by score
    results.sort(key=lambda r: r.overall_score, reverse=True)

    # Assign sequential ranks
    for idx, r in enumerate(results):
        r.rank = idx + 1

    return results
