"""
Level 1: Simple Wind-Driven Backward Drift Model
------------------------------------------------
Computes backwards trajectory integration using windage velocity:
    V_drift = alpha * V_wind
    Position(t - dt) = Position(t) - V_drift * dt

SCIENTIFIC LIMITATION NOTICE:
This is an experimental baseline. It does NOT account for ocean currents, waves,
or oil weathering. True ocean transport is dominated by surface currents.
"""

from datetime import datetime, timedelta
from typing import Optional, List
import math

from src.ais.schema import SpillEvent
from src.hindcasting.schema import HindcastResult, TrajectoryPoint, EnvironmentalVector
from src.hindcasting.interface import HindcastingModelInterface
from src.hindcasting.environmental import EnvironmentalDataProvider, ConstantEnvironmentalProvider
from src.hindcasting.coordinates import displace_lat_lon, haversine_km


class SimpleWindDriftModel(HindcastingModelInterface):
    """
    Level 1 Model:
    Simulates backward drift due to surface wind friction using a configurable windage factor (default 3%).
    """
    def __init__(
        self,
        windage_factor: float = 0.030,
        time_step_hours: float = 1.0,
        base_uncertainty_km: float = 5.0,
        uncertainty_rate_km_per_hr: float = 0.5
    ):
        """
        Args:
            windage_factor: Fraction of 10m wind transferred to surface slick (typically 0.025 - 0.035).
            time_step_hours: Discrete integration step size in hours.
            base_uncertainty_km: Initial detection uncertainty in km.
            uncertainty_rate_km_per_hr: Uncertainty expansion per hour of backward simulation.
        """
        self.windage_factor = windage_factor
        self.time_step_hours = time_step_hours
        self.base_uncertainty_km = base_uncertainty_km
        self.uncertainty_rate = uncertainty_rate_km_per_hr

    @property
    def model_name(self) -> str:
        return "SimpleWindDriftModel"

    @property
    def model_level(self) -> int:
        return 1

    def estimate_origin(
        self,
        spill: SpillEvent,
        drift_hours: float = 12.0,
        env_provider: Optional[EnvironmentalDataProvider] = None
    ) -> HindcastResult:
        warnings = []
        assumptions = [
            f"Windage coefficient = {self.windage_factor * 100:.1f}%",
            "Zero ocean current influence assumed (Level 1 simplification)",
            "Backward Euler integration used",
            "Linear spatial-temporal wind field"
        ]

        if env_provider is None:
            # Fallback to zero wind if no provider supplied
            warnings.append("No environmental data provider provided. Drift velocity = 0.")
            env_provider = ConstantEnvironmentalProvider(
                wind=EnvironmentalVector(u=0.0, v=0.0, source="fallback_zero")
            )

        obs_time = spill.observation_time
        curr_lat = spill.centroid_lat
        curr_lon = spill.centroid_lon
        curr_time = obs_time

        trajectory: List[TrajectoryPoint] = []
        cumulative_dist = 0.0

        # Step 0: Observation point (t = 0)
        trajectory.append(TrajectoryPoint(
            timestamp=curr_time,
            step_hours_from_obs=0.0,
            latitude=curr_lat,
            longitude=curr_lon,
            u_velocity_ms=0.0,
            v_velocity_ms=0.0,
            cumulative_drift_km=0.0
        ))

        n_steps = int(math.ceil(drift_hours / self.time_step_hours))
        dt_seconds = self.time_step_hours * 3600.0

        for step in range(1, n_steps + 1):
            # Query wind at current location and time
            wind = env_provider.get_wind(curr_lat, curr_lon, curr_time)
            if wind is None:
                warnings.append(f"Wind data unavailable at step {step} ({curr_time.isoformat()}).")
                u_wind, v_wind = 0.0, 0.0
            else:
                u_wind, v_wind = wind.u, wind.v

            # Forward drift velocity = windage * wind
            # Backward movement = - forward drift velocity
            u_drift = self.windage_factor * u_wind
            v_drift = self.windage_factor * v_wind

            # Negative displacement for backward integration: dx = - u_drift * dt
            dx_meters = - u_drift * dt_seconds
            dy_meters = - v_drift * dt_seconds

            prev_lat, prev_lon = displace_lat_lon(curr_lat, curr_lon, dx_meters, dy_meters)
            
            step_km = haversine_km(curr_lat, curr_lon, prev_lat, prev_lon)
            cumulative_dist += step_km

            curr_lat = prev_lat
            curr_lon = prev_lon
            curr_time = curr_time - timedelta(hours=self.time_step_hours)

            trajectory.append(TrajectoryPoint(
                timestamp=curr_time,
                step_hours_from_obs=-(step * self.time_step_hours),
                latitude=curr_lat,
                longitude=curr_lon,
                u_velocity_ms=u_drift,
                v_velocity_ms=v_drift,
                cumulative_drift_km=round(cumulative_dist, 3)
            ))

        # Release origin is the endpoint of the backward trajectory
        origin_lat = curr_lat
        origin_lon = curr_lon

        # Release time window: from earliest backward time to observation time
        origin_time_start = curr_time
        origin_time_end = obs_time

        # Uncertainty grows with duration
        origin_uncertainty_km = self.base_uncertainty_km + (self.uncertainty_rate * drift_hours)

        return HindcastResult(
            spill_id=spill.spill_id,
            observed_lat=spill.centroid_lat,
            observed_lon=spill.centroid_lon,
            observation_time=obs_time,
            timestamp_precision="exact",
            estimated_origin_lat=round(origin_lat, 5),
            estimated_origin_lon=round(origin_lon, 5),
            origin_time_start=origin_time_start,
            origin_time_end=origin_time_end,
            drift_duration_hours=drift_hours,
            origin_uncertainty_km=round(origin_uncertainty_km, 2),
            trajectory=trajectory,
            model_name=self.model_name,
            model_level=self.model_level,
            environmental_sources={"wind": getattr(wind, "source", "provided")},
            assumptions=assumptions,
            warnings=warnings,
            metadata={
                "windage_factor": self.windage_factor,
                "time_step_hours": self.time_step_hours,
                "total_drift_distance_km": round(cumulative_dist, 3)
            }
        )
