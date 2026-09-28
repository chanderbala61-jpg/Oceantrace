"""
Standard Internal Data Schemas for OceanTrace AIS Correlation Module
--------------------------------------------------------------------
Defines:
  - AISRecord: Single transponder broadcast ping
  - SpillEvent: Detected SAR spill characterization + origin metadata
  - CandidateVesselResult: Ranked candidate vessel correlation output
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


@dataclass(frozen=True)
class AISRecord:
    """
    Standard schema for a single AIS vessel broadcast ping.
    Tolerates missing optional fields while enforcing spatial and temporal bounds.
    """
    timestamp: datetime
    mmsi: int
    latitude: float
    longitude: float
    sog: float
    cog: float
    heading: Optional[float] = None
    nav_status: Optional[int] = None
    
    # Optional vessel voyage / registry metadata
    ship_name: Optional[str] = None
    ship_type: Optional[str] = None
    imo: Optional[int] = None
    length: Optional[float] = None
    width: Optional[float] = None
    draught: Optional[float] = None


@dataclass
class SpillEvent:
    """
    Standard representation of an oil spill detection event.
    
    Distinguishes between:
      - Observed detection state (SAR observation time, observed centroid, polygon)
      - Estimated release state (hindcasted origin coordinates, origin time window)
    """
    spill_id: str
    observation_time: datetime
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    polygon: Optional[List[Tuple[float, float]]] = None
    observation_time_end: Optional[datetime] = None

    # Hindcasted origin fields (None if hindcasting has not yet been executed)
    estimated_origin_lat: Optional[float] = None
    estimated_origin_lon: Optional[float] = None
    origin_time_start: Optional[datetime] = None
    origin_time_end: Optional[datetime] = None
    origin_uncertainty_km: float = 10.0

    @property
    def effective_origin_lat(self) -> float:
        """Returns estimated origin lat if available, else observed centroid lat."""
        return self.estimated_origin_lat if self.estimated_origin_lat is not None else self.centroid_lat

    @property
    def effective_origin_lon(self) -> float:
        """Returns estimated origin lon if available, else observed centroid lon."""
        return self.estimated_origin_lon if self.estimated_origin_lon is not None else self.centroid_lon

    @property
    def has_hindcast_origin(self) -> bool:
        """Indicates whether backwards drift hindcasting has been applied."""
        return self.estimated_origin_lat is not None and self.estimated_origin_lon is not None


@dataclass
class CandidateVesselResult:
    """
    Represents an explainable evaluation of a candidate vessel's correlation with a spill.
    
    IMPORTANT POLICY:
    Never claims 'This vessel caused the spill'.
    Uses compatibility classes ('Highly Compatible', 'Moderately Compatible', 'Low Compatibility').
    """
    mmsi: int
    rank: int
    overall_score: float  # Scale: 0.0 to 1.0
    classification: str   # 'Highly Compatible Candidate', 'Moderately Compatible Candidate', etc.
    
    # Individual evidence factor scores [0.0, 1.0]
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Quantitative measurements
    measurements: Dict[str, Any] = field(default_factory=dict)
    
    # Natural language explanation of the score
    explanation: str = ""
    
    # Audit trail documenting why candidate passed or failed various filters
    audit_trail: Dict[str, Any] = field(default_factory=dict)

    @property
    def scoring_breakdown(self) -> Dict[str, float]:
        """Provides mapped dictionary of individual component scores for UI display."""
        ev = self.evidence_scores if isinstance(self.evidence_scores, dict) else {}
        spat = float(ev.get("spatial_score", ev.get("spatial_proximity", 0.0)))
        temp = float(ev.get("temporal_score", ev.get("temporal_alignment", 0.0)))
        traj = float(ev.get("trajectory_score", ev.get("trajectory_overlap", 0.0)))
        kin = float(ev.get("kinematic_score", ev.get("kinematic_consistency", 0.0)))
        return {
            "spatial_score": spat,
            "temporal_score": temp,
            "trajectory_score": traj,
            "kinematic_score": kin,
            "spatial_proximity": spat,
            "temporal_alignment": temp,
            "trajectory_overlap": traj,
            "kinematic_consistency": kin,
        }
