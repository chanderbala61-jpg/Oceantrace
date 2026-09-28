"""
Data Schemas for OceanTrace Hindcasting Engine
----------------------------------------------
Defines:
  - TrajectoryPoint: Single point along the backward drift track
  - HindcastResult: Full output contract of the backward hindcast simulation
  - EnvironmentalVector: Vector components for wind or surface currents
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any


@dataclass(frozen=True)
class EnvironmentalVector:
    """
    Vector components of wind or ocean surface currents.
    u: Eastward velocity component (m/s)
    v: Northward velocity component (m/s)
    """
    u: float
    v: float
    source: str = "unknown"
    confidence: float = 1.0

    @property
    def speed(self) -> float:
        import math
        return math.sqrt(self.u ** 2 + self.v ** 2)

    @property
    def direction_to_degrees(self) -> float:
        """Compass direction TO which the vector points [0, 360)."""
        import math
        deg = math.degrees(math.atan2(self.u, self.v))
        return (deg + 360.0) % 360.0


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    Single discrete point along the backwards hindcast path.
    """
    timestamp: datetime
    step_hours_from_obs: float  # e.g., 0.0, -1.0, -2.0, ... -drift_hours
    latitude: float
    longitude: float
    u_velocity_ms: float        # Total drift velocity u component (m/s)
    v_velocity_ms: float        # Total drift velocity v component (m/s)
    cumulative_drift_km: float

    @property
    def lat(self) -> float:
        return self.latitude

    @property
    def lon(self) -> float:
        return self.longitude

    @property
    def time_offset_hours(self) -> float:
        return abs(self.step_hours_from_obs)

    @property
    def uncertainty_radius_km(self) -> float:
        return max(1.0, abs(self.step_hours_from_obs) * 0.5)


@dataclass
class HindcastResult:
    """
    Standard output container for oil-spill backward hindcast simulations.
    
    Contains:
      - Observed detection coordinates and observation timestamp
      - Timestamp precision ('exact' vs 'date_only')
      - Estimated release coordinates and release time window
      - Discrete backwards trajectory points
      - Model metadata, environmental data sources, and explicit scientific limitations/warnings
    """
    spill_id: str
    observed_lat: float
    observed_lon: float
    observation_time: datetime
    timestamp_precision: str    # "exact" or "date_only"
    
    # Estimated historical release state
    estimated_origin_lat: float
    estimated_origin_lon: float
    origin_time_start: datetime
    origin_time_end: datetime
    drift_duration_hours: float
    origin_uncertainty_km: float
    
    # Backwards trajectory track
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    
    # Model and provenance audit metadata
    model_name: str = "unspecified"
    model_level: int = 0
    environmental_sources: Dict[str, str] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
