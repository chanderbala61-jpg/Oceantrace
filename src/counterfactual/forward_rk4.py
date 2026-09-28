"""
Forward 4th-Order Runge-Kutta (RK4) Drift Simulation Engine
===========================================================
Computes forward-in-time Lagrangian trajectory integration from a hypothetical
spill release position up to the target satellite observation timestamp:

    dx/dt = + (u_current + alpha_wind * u_wind)
    dy/dt = + (v_current + alpha_wind * v_wind)

Integration Method:
    4th-Order Runge-Kutta (RK4) with metric displacement mapped via WGS-84 geodesic
    transforms (displace_lat_lon).

SCIENTIFIC INTEGRITY NOTICE:
    This is an advective drift model with expanding Gaussian dispersion. It does NOT
    simulate complex chemical weathering, evaporation, emulsification, or wave-breaking
    dispersion, as those require calibrated operational packages (e.g. NOAA GNOME, OpenDrift).
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

from src.hindcasting.schema import TrajectoryPoint, EnvironmentalVector
from src.hindcasting.environmental import EnvironmentalDataProvider, ConstantEnvironmentalProvider
from src.hindcasting.coordinates import displace_lat_lon, haversine_km
from src.counterfactual.schema import HypotheticalRelease, ForwardSimulationConfig


class ForwardRK4DriftModel:
    """
    Forward Lagrangian Drift Model employing 4th-Order Runge-Kutta numerical integration.
    """
    def __init__(self, config: Optional[ForwardSimulationConfig] = None):
        self.config = config or ForwardSimulationConfig()

    @property
    def model_name(self) -> str:
        return "ForwardLagrangianRK4Drift"

    def _get_drift_velocity(
        self,
        lat: float,
        lon: float,
        timestamp: datetime,
        env_provider: EnvironmentalDataProvider
    ) -> Tuple[float, float, str]:
        """
        Calculates total instantaneous forward surface drift velocity components (u, v) in m/s:
            u_drift = u_current + windage * u_wind
            v_drift = v_current + windage * v_wind
        """
        curr = env_provider.get_current(lat, lon, timestamp)
        wind = env_provider.get_wind(lat, lon, timestamp)

        u_c = curr.u if curr is not None else 0.0
        v_c = curr.v if curr is not None else 0.0

        u_w = wind.u if wind is not None else 0.0
        v_w = wind.v if wind is not None else 0.0

        u_drift = u_c + (self.config.windage_factor * u_w)
        v_drift = v_c + (self.config.windage_factor * v_w)

        src_info = f"Currents: {curr.source if curr else 'none'}, Wind: {wind.source if wind else 'none'}"
        return u_drift, v_drift, src_info

    def simulate_forward(
        self,
        release: HypotheticalRelease,
        target_observation_time: datetime,
        env_provider: Optional[EnvironmentalDataProvider] = None,
        max_duration_hours: Optional[float] = None
    ) -> Tuple[List[TrajectoryPoint], List[str], List[str]]:
        """
        Simulates the forward drift path from release.release_time up to target_observation_time.

        Args:
            release: HypotheticalRelease event.
            target_observation_time: Datetime of satellite SAR observation.
            env_provider: EnvironmentalDataProvider supplying current and wind vectors.
            max_duration_hours: Optional duration cap.

        Returns:
            (trajectory, assumptions, warnings)
        """
        assumptions = [
            "Forward 4th-Order Runge-Kutta (RK4) numerical integration",
            f"Windage coefficient = {self.config.windage_factor * 100:.1f}% of 10m wind velocity",
            "Superposition of ocean surface currents and atmospheric wind drag",
            f"Discrete integration time step = {self.config.time_step_hours * 60:.0f} minutes",
            "No chemical weathering or biodegradation modeled (advection-diffusion prototype)"
        ]
        warnings = []

        if env_provider is None:
            warnings.append("No environmental data provider supplied. Simulating with zero drift velocity.")
            env_provider = ConstantEnvironmentalProvider(
                wind=EnvironmentalVector(u=0.0, v=0.0, source="fallback_zero"),
                current=EnvironmentalVector(u=0.0, v=0.0, source="fallback_zero")
            )

        start_time = release.release_time
        if target_observation_time < start_time:
            warnings.append(
                f"Observation time ({target_observation_time.isoformat()}) is earlier than release time "
                f"({start_time.isoformat()}). Clamping simulation duration to 0.0 hours."
            )
            total_duration_hours = 0.0
        else:
            total_duration_hours = (target_observation_time - start_time).total_seconds() / 3600.0

        max_dur = max_duration_hours or self.config.max_drift_hours
        if total_duration_hours > max_dur:
            warnings.append(
                f"Simulation duration ({total_duration_hours:.1f}h) exceeds safety ceiling ({max_dur:.1f}h); capping."
            )
            total_duration_hours = max_dur

        curr_lat = float(release.release_lat)
        curr_lon = float(release.release_lon)
        curr_time = start_time
        cumulative_km = 0.0

        trajectory: List[TrajectoryPoint] = []

        # Step 0: Initial hypothetical release point
        trajectory.append(TrajectoryPoint(
            timestamp=curr_time,
            step_hours_from_obs=-(total_duration_hours),
            latitude=curr_lat,
            longitude=curr_lon,
            u_velocity_ms=0.0,
            v_velocity_ms=0.0,
            cumulative_drift_km=0.0
        ))

        if total_duration_hours <= 0.0:
            return trajectory, assumptions, warnings

        step_hours = self.config.time_step_hours
        dt_seconds = step_hours * 3600.0
        elapsed_hours = 0.0

        while elapsed_hours < total_duration_hours - 1e-6:
            # Handle final partial step if total_duration_hours is not an exact multiple
            actual_step_hours = min(step_hours, total_duration_hours - elapsed_hours)
            actual_dt_seconds = actual_step_hours * 3600.0

            # ----------------------------------------------------
            # 4th-Order Runge-Kutta Integration (Forward in Time)
            # ----------------------------------------------------
            # k1 at (t_n, x_n)
            u1, v1, _ = self._get_drift_velocity(curr_lat, curr_lon, curr_time, env_provider)

            # k2 at (t_n + dt/2, x_n + k1 * dt/2)
            dx1 = u1 * (actual_dt_seconds * 0.5)
            dy1 = v1 * (actual_dt_seconds * 0.5)
            lat_k2, lon_k2 = displace_lat_lon(curr_lat, curr_lon, dx1, dy1)
            t_half = curr_time + timedelta(seconds=actual_dt_seconds * 0.5)
            u2, v2, _ = self._get_drift_velocity(lat_k2, lon_k2, t_half, env_provider)

            # k3 at (t_n + dt/2, x_n + k2 * dt/2)
            dx2 = u2 * (actual_dt_seconds * 0.5)
            dy2 = v2 * (actual_dt_seconds * 0.5)
            lat_k3, lon_k3 = displace_lat_lon(curr_lat, curr_lon, dx2, dy2)
            u3, v3, _ = self._get_drift_velocity(lat_k3, lon_k3, t_half, env_provider)

            # k4 at (t_n + dt, x_n + k3 * dt)
            dx3 = u3 * actual_dt_seconds
            dy3 = v3 * actual_dt_seconds
            lat_k4, lon_k4 = displace_lat_lon(curr_lat, curr_lon, dx3, dy3)
            t_full = curr_time + timedelta(seconds=actual_dt_seconds)
            u4, v4, _ = self._get_drift_velocity(lat_k4, lon_k4, t_full, env_provider)

            # Weighted RK4 composite velocity
            u_rk4 = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
            v_rk4 = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

            # Forward displacement
            step_dx_meters = u_rk4 * actual_dt_seconds
            step_dy_meters = v_rk4 * actual_dt_seconds

            next_lat, next_lon = displace_lat_lon(curr_lat, curr_lon, step_dx_meters, step_dy_meters)
            step_dist_km = haversine_km(curr_lat, curr_lon, next_lat, next_lon)
            cumulative_km += step_dist_km

            elapsed_hours += actual_step_hours
            curr_time = curr_time + timedelta(seconds=actual_dt_seconds)
            curr_lat = next_lat
            curr_lon = next_lon

            # Remaining offset from target observation
            remaining_hours_to_obs = -(total_duration_hours - elapsed_hours)

            trajectory.append(TrajectoryPoint(
                timestamp=curr_time,
                step_hours_from_obs=round(remaining_hours_to_obs, 3),
                latitude=curr_lat,
                longitude=curr_lon,
                u_velocity_ms=round(u_rk4, 4),
                v_velocity_ms=round(v_rk4, 4),
                cumulative_drift_km=round(cumulative_km, 3)
            ))

        return trajectory, assumptions, warnings
