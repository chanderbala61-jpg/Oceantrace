"""
Level 0: Direct Observation Baseline Model
------------------------------------------
Returns observed centroid as origin with explicit scientific disclaimers.
Used when environmental hindcasting data is unavailable.
"""

from datetime import datetime, timedelta
from typing import Optional
from src.ais.schema import SpillEvent
from src.hindcasting.schema import HindcastResult, TrajectoryPoint
from src.hindcasting.interface import HindcastingModelInterface
from src.hindcasting.environmental import EnvironmentalDataProvider


class DirectObservationBaseline(HindcastingModelInterface):
    """
    Level 0 Baseline:
    Assumes estimated origin equals observed centroid.
    Does not compute backward trajectory.
    """
    @property
    def model_name(self) -> str:
        return "DirectObservationBaseline"

    @property
    def model_level(self) -> int:
        return 0

    def estimate_origin(
        self,
        spill: SpillEvent,
        drift_hours: float = 12.0,
        env_provider: Optional[EnvironmentalDataProvider] = None
    ) -> HindcastResult:
        obs_time = spill.observation_time
        
        # Origin time window spans back drift_hours
        t_start = obs_time - timedelta(hours=drift_hours)
        t_end = obs_time

        # Single point trajectory (t=0)
        p0 = TrajectoryPoint(
            timestamp=obs_time,
            step_hours_from_obs=0.0,
            latitude=spill.centroid_lat,
            longitude=spill.centroid_lon,
            u_velocity_ms=0.0,
            v_velocity_ms=0.0,
            cumulative_drift_km=0.0
        )

        warning_msg = (
            "Environmental backward drift data unavailable. "
            "Observed SAR centroid used as baseline origin."
        )

        return HindcastResult(
            spill_id=spill.spill_id,
            observed_lat=spill.centroid_lat,
            observed_lon=spill.centroid_lon,
            observation_time=obs_time,
            timestamp_precision="exact",
            estimated_origin_lat=spill.centroid_lat,
            estimated_origin_lon=spill.centroid_lon,
            origin_time_start=t_start,
            origin_time_end=t_end,
            drift_duration_hours=drift_hours,
            origin_uncertainty_km=max(spill.origin_uncertainty_km, 15.0),
            trajectory=[p0],
            model_name=self.model_name,
            model_level=self.model_level,
            environmental_sources={"wind": "none", "current": "none"},
            assumptions=["Zero historical drift assumed", "Stationary slick baseline"],
            warnings=[warning_msg]
        )
