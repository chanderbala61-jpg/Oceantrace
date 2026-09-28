"""
Counterfactual Vessel Testing Data Schemas
==========================================
Defines:
  - HypotheticalRelease: Hypothetical spill release event tied to an AIS candidate position
  - ForwardSimulationConfig: Parameters for forward 4th-Order Runge-Kutta numerical integration
  - ConsistencyDiagnostics: Quantifiable geometric and temporal comparisons with observed SAR slick
  - CounterfactualResult: Complete audit trail container for candidate counterfactual evaluation

SCIENTIFIC INTEGRITY GUARDRAIL:
  Hypothetical releases do NOT state or imply that the vessel discharged oil.
  Evaluations reflect spatiotemporal trajectory consistency, not legal culpability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.hindcasting.schema import TrajectoryPoint


@dataclass(frozen=True)
class HypotheticalRelease:
    """
    Standard container defining a hypothetical release event at an AIS vessel position.
    
    POLICY: Strictly labeled as a hypothetical test event for diagnostic modeling.
    """
    mmsi: int
    vessel_name: str
    vessel_type: str
    release_lat: float
    release_lon: float
    release_time: datetime
    sog_knots: Optional[float] = None
    cog_deg: Optional[float] = None
    heading_deg: Optional[float] = None
    cpa_distance_km: Optional[float] = None
    prior_ais_score: float = 0.0
    source_notes: str = "Actual AIS transponder ping selected as hypothetical release origin."
    label: str = "Hypothetical Release"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
            "release_lat": self.release_lat,
            "release_lon": self.release_lon,
            "release_time": self.release_time.isoformat(),
            "sog_knots": self.sog_knots,
            "cog_deg": self.cog_deg,
            "heading_deg": self.heading_deg,
            "cpa_distance_km": self.cpa_distance_km,
            "prior_ais_score": self.prior_ais_score,
            "label": self.label,
        }


@dataclass
class ForwardSimulationConfig:
    """
    Configuration parameters for forward Lagrangian drift simulation.
    """
    windage_factor: float = 0.030          # 3.0% standard surface windage
    time_step_hours: float = 0.5           # Discrete integration time step in hours
    initial_footprint_radius_km: float = 0.5  # Initial near-field release zone radius (km)
    diffusion_rate_km_sqrt_hr: float = 0.4    # Turbulent Gaussian diffusion spread rate
    max_drift_hours: float = 48.0          # Maximum allowable simulation duration


@dataclass
class ConsistencyDiagnostics:
    """
    Deterministic quantitative comparisons between forward simulated slick and observed SAR detection.
    
    IMPORTANT: These are geometric consistency measures, NOT a probability of responsibility.
    """
    # Spatial Endpoints
    simulated_endpoint_lat: float
    simulated_endpoint_lon: float
    simulated_footprint_radius_km: float
    simulated_arrival_time: datetime
    observed_centroid_lat: float
    observed_centroid_lon: float
    observed_time: datetime

    # Measurable Geometric Metrics
    centroid_distance_km: float            # Distance from simulated endpoint to observed SAR centroid
    trajectory_min_distance_to_origin_km: float  # Min distance between forward path and hindcast origin
    temporal_offset_hours: float           # Duration between hypothetical release and observation

    # Diagnostic Overlap & Containment
    footprint_containment: bool            # True if observed centroid is within simulated footprint radius
    spatial_proximity_score: float         # [0.0, 1.0] exponential decay with centroid distance
    trajectory_proximity_score: float      # [0.0, 1.0] exponential decay with origin proximity
    temporal_alignment_score: float        # [0.0, 1.0] consistency with drift time window
    prior_ais_score: float                 # [0.0, 1.0] baseline multi-factor AIS association score

    # Combined Deterministic Prioritization Score
    candidate_prioritization_score: float  # [0.0, 1.0] weighted composite score
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    classification: str = "Low Counterfactual Consistency"
    explanation: str = ""
    diagnostic_notes: List[str] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """
    Full output container for an AIS Candidate Counterfactual Drift Test.
    """
    candidate_mmsi: int
    vessel_name: str
    hypothetical_release: HypotheticalRelease
    config: ForwardSimulationConfig
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    diagnostics: Optional[ConsistencyDiagnostics] = None
    environmental_sources: Dict[str, str] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_trajectory(self) -> bool:
        return len(self.trajectory) > 0

    @property
    def endpoint(self) -> Optional[TrajectoryPoint]:
        return self.trajectory[-1] if self.trajectory else None
