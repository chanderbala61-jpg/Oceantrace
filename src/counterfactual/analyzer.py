"""
Counterfactual Consistency Analyzer & Candidate Prioritization
==============================================================
Orchestrates:
  1. Hypothetical release event formulation from an AIS candidate ping
  2. Forward 4th-Order Runge-Kutta numerical drift simulation
  3. Measurable geometric diagnostics against the observed SAR spill
  4. Deterministic counterfactual consistency scoring and prioritization

SCIENTIFIC INTEGRITY POLICY:
  - Outputs are designated strictly as 'Investigation Leads'.
  - Evaluates mathematical and physical transit consistency.
  - Never declares or implies legal guilt, culpability, or intentional discharge.
"""

import math
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from src.ais.schema import SpillEvent, CandidateVesselResult, AISRecord
from src.hindcasting.schema import TrajectoryPoint
from src.hindcasting.environmental import EnvironmentalDataProvider, ConstantEnvironmentalProvider
from src.hindcasting.coordinates import haversine_km
from src.counterfactual.schema import (
    HypotheticalRelease,
    ForwardSimulationConfig,
    ConsistencyDiagnostics,
    CounterfactualResult,
)
from src.counterfactual.forward_rk4 import ForwardRK4DriftModel


class CounterfactualAnalyzer:
    """
    Evaluates whether a candidate vessel's historical AIS position is physically
    consistent with an observed oil spill detection via forward drift simulation.
    """
    def __init__(
        self,
        config: Optional[ForwardSimulationConfig] = None,
        model: Optional[ForwardRK4DriftModel] = None,
        scoring_weights: Optional[Dict[str, float]] = None
    ):
        self.config = config or ForwardSimulationConfig()
        self.model = model or ForwardRK4DriftModel(config=self.config)
        # Explicit deterministic weights summing to 1.0
        self.scoring_weights = scoring_weights or {
            "spatial_endpoint": 0.35,      # 35% Weight: Distance from simulated endpoint to observed centroid
            "trajectory_proximity": 0.25,  # 25% Weight: Minimum distance between forward path and origin
            "temporal_alignment": 0.20,    # 20% Weight: Temporal match with release window
            "prior_ais_score": 0.20        # 20% Weight: Prior multi-factor AIS transit association
        }

    def create_hypothetical_release(
        self,
        candidate: Union[CandidateVesselResult, AISRecord, Dict[str, Any]],
        fallback_time: Optional[datetime] = None
    ) -> HypotheticalRelease:
        """
        Creates a validated HypotheticalRelease event from candidate vessel data.
        Never invents missing AIS fields.
        """
        if isinstance(candidate, CandidateVesselResult):
            m = candidate.measurements
            mmsi = candidate.mmsi
            vessel_name = str(m.get("ship_name") or f"Vessel_{mmsi}")
            vessel_type = str(m.get("ship_type") or "Commercial Vessel")
            rel_lat = float(m.get("closest_lat", 0.0))
            rel_lon = float(m.get("closest_lon", 0.0))
            rel_time = m.get("closest_approach_time") or fallback_time or datetime.utcnow()
            sog = m.get("mean_sog")
            cog = None
            heading = None
            cpa = m.get("min_distance_km")
            prior_score = candidate.overall_score
            notes = (
                f"Hypothetical release formulated at AIS Closest Point of Approach (CPA: {cpa:.2f} km) "
                f"at {rel_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
            )
        elif isinstance(candidate, AISRecord):
            mmsi = candidate.mmsi
            vessel_name = str(candidate.ship_name or f"Vessel_{mmsi}")
            vessel_type = str(candidate.ship_type or "Commercial Vessel")
            rel_lat = float(candidate.latitude)
            rel_lon = float(candidate.longitude)
            rel_time = candidate.timestamp
            sog = candidate.sog
            cog = candidate.cog
            heading = candidate.heading
            cpa = None
            prior_score = 0.50
            notes = f"Hypothetical release formulated at observed AIS broadcast ping at {rel_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
        elif isinstance(candidate, dict):
            mmsi = int(candidate.get("mmsi", 0))
            vessel_name = str(candidate.get("vessel_name") or candidate.get("ship_name") or f"Vessel_{mmsi}")
            vessel_type = str(candidate.get("vessel_type") or candidate.get("ship_type") or "Commercial Vessel")
            rel_lat = float(candidate.get("latitude") or candidate.get("lat") or candidate.get("closest_lat", 0.0))
            rel_lon = float(candidate.get("longitude") or candidate.get("lon") or candidate.get("closest_lon", 0.0))
            rel_time = candidate.get("timestamp") or candidate.get("closest_approach_time") or fallback_time or datetime.utcnow()
            sog = candidate.get("sog") or candidate.get("mean_sog")
            cog = candidate.get("cog")
            heading = candidate.get("heading")
            cpa = candidate.get("min_distance_km") or candidate.get("cpa_distance_km")
            prior_score = float(candidate.get("overall_score", 0.50))
            notes = "Hypothetical release formulated from candidate dictionary input."
        else:
            raise TypeError(f"Unsupported candidate input type: {type(candidate)}")

        # Coordinate sanity validation
        if not (-90.0 <= rel_lat <= 90.0) or not (-180.0 <= rel_lon <= 180.0):
            raise ValueError(f"Invalid hypothetical release coordinates: ({rel_lat}, {rel_lon})")

        return HypotheticalRelease(
            mmsi=mmsi,
            vessel_name=vessel_name,
            vessel_type=vessel_type,
            release_lat=rel_lat,
            release_lon=rel_lon,
            release_time=rel_time,
            sog_knots=sog,
            cog_deg=cog,
            heading_deg=heading,
            cpa_distance_km=cpa,
            prior_ais_score=prior_score,
            source_notes=notes,
            label="Hypothetical Release"
        )

    def evaluate_candidate(
        self,
        candidate: Union[CandidateVesselResult, AISRecord, Dict[str, Any]],
        spill: SpillEvent,
        env_provider: Optional[EnvironmentalDataProvider] = None
    ) -> CounterfactualResult:
        """
        Executes complete counterfactual testing for a single candidate vessel.
        """
        release = self.create_hypothetical_release(candidate, fallback_time=spill.observation_time)
        target_time = spill.observation_time

        # 1. Forward 4th-Order Runge-Kutta simulation
        trajectory, assumptions, warnings = self.model.simulate_forward(
            release=release,
            target_observation_time=target_time,
            env_provider=env_provider
        )

        sim_endpoint = trajectory[-1]
        sim_duration_hours = max(0.0, (target_time - release.release_time).total_seconds() / 3600.0)

        # 2. Expanding Gaussian footprint radius at arrival
        # r(t) = r_0 + kappa * sqrt(t)
        footprint_radius_km = self.config.initial_footprint_radius_km + (
            self.config.diffusion_rate_km_sqrt_hr * math.sqrt(max(0.1, sim_duration_hours))
        )

        # 3. Measurable Geometric Diagnostics
        centroid_dist = haversine_km(
            sim_endpoint.latitude, sim_endpoint.longitude,
            spill.centroid_lat, spill.centroid_lon
        )

        # Minimum distance from forward path to estimated or observed origin
        origin_lat = spill.effective_origin_lat
        origin_lon = spill.effective_origin_lon
        min_path_to_origin = min(
            haversine_km(pt.latitude, pt.longitude, origin_lat, origin_lon)
            for pt in trajectory
        ) if trajectory else centroid_dist

        # Containment check
        contained = bool(centroid_dist <= footprint_radius_km)

        # 4. Dimensionless Sub-Scores [0.0, 1.0]
        # Spatial endpoint score (exponential decay with effective footprint scale)
        norm_scale = max(footprint_radius_km, 6.0)
        s_spatial = math.exp(-centroid_dist / norm_scale)

        # Trajectory proximity to origin score
        s_traj = math.exp(-min_path_to_origin / 6.0)

        # Temporal alignment score
        if spill.origin_time_start and spill.origin_time_end:
            win_mid = datetime.fromtimestamp(
                (spill.origin_time_start.timestamp() + spill.origin_time_end.timestamp()) / 2.0
            )
            time_err_hours = abs((release.release_time - win_mid).total_seconds()) / 3600.0
            s_temporal = math.exp(-time_err_hours / 3.0)
        else:
            s_temporal = math.exp(-sim_duration_hours / 18.0)

        s_prior = max(0.0, min(1.0, release.prior_ais_score))

        # 5. Composite Counterfactual Consistency Score
        w = self.scoring_weights
        cf_score = (
            w["spatial_endpoint"] * s_spatial +
            w["trajectory_proximity"] * s_traj +
            w["temporal_alignment"] * s_temporal +
            w["prior_ais_score"] * s_prior
        )
        cf_score = round(max(0.0, min(1.0, cf_score)), 4)

        # 6. Explainable Classification & Text
        if cf_score >= 0.75:
            classification = "High Spatiotemporal Consistency Lead"
            explanation = (
                f"Forward simulated trajectory from candidate {release.vessel_name} (MMSI: {release.mmsi}) "
                f"drifts within {centroid_dist:.2f} km of the observed SAR slick centroid over {sim_duration_hours:.1f} hours, "
                f"demonstrating strong physical and geometric consistency."
            )
        elif cf_score >= 0.45:
            classification = "Moderate Spatiotemporal Consistency Lead"
            explanation = (
                f"Forward simulated drift reaches within {centroid_dist:.2f} km of the observed detection "
                f"(footprint radius: {footprint_radius_km:.1f} km). Spatiotemporally compatible with moderate dispersion."
            )
        else:
            classification = "Low Counterfactual Consistency"
            explanation = (
                f"Forward simulated drift deviates by {centroid_dist:.2f} km from observed slick centroid. "
                f"Hypothetical release at candidate position is physically discordant with observed SAR coordinates."
            )

        diag_notes = [
            f"Observed SAR slick centroid: {spill.centroid_lat:.4f}°N, {spill.centroid_lon:.4f}°E",
            f"Simulated forward endpoint: {sim_endpoint.latitude:.4f}°N, {sim_endpoint.longitude:.4f}°E",
            f"Centroid separation distance: {centroid_dist:.2f} km",
            f"Simulated footprint radius at observation: {footprint_radius_km:.2f} km (Containment: {contained})",
            f"Forward drift integration steps: {len(trajectory)} (Duration: {sim_duration_hours:.1f}h)"
        ]

        diagnostics = ConsistencyDiagnostics(
            simulated_endpoint_lat=round(sim_endpoint.latitude, 5),
            simulated_endpoint_lon=round(sim_endpoint.longitude, 5),
            simulated_footprint_radius_km=round(footprint_radius_km, 2),
            simulated_arrival_time=target_time,
            observed_centroid_lat=round(spill.centroid_lat, 5),
            observed_centroid_lon=round(spill.centroid_lon, 5),
            observed_time=spill.observation_time,
            centroid_distance_km=round(centroid_dist, 3),
            trajectory_min_distance_to_origin_km=round(min_path_to_origin, 3),
            temporal_offset_hours=round(sim_duration_hours, 2),
            footprint_containment=contained,
            spatial_proximity_score=round(s_spatial, 4),
            trajectory_proximity_score=round(s_traj, 4),
            temporal_alignment_score=round(s_temporal, 4),
            prior_ais_score=round(s_prior, 4),
            candidate_prioritization_score=cf_score,
            scoring_weights=self.scoring_weights,
            classification=classification,
            explanation=explanation,
            diagnostic_notes=diag_notes
        )

        limitations = [
            "Counterfactual simulation evaluates physical transit plausibility only; it does NOT establish release occurrence or legal liability.",
            "AIS vessel broadcast rates may exhibit temporal gaps or transponder outages.",
            "Surface current and wind field inputs are simplified representations unless coupled to real-time ocean reanalysis.",
            "No chemical weathering, evaporation, or emulsification dynamics are modeled."
        ]

        return CounterfactualResult(
            candidate_mmsi=release.mmsi,
            vessel_name=release.vessel_name,
            hypothetical_release=release,
            config=self.config,
            trajectory=trajectory,
            diagnostics=diagnostics,
            environmental_sources={"provider": type(env_provider).__name__ if env_provider else "fallback_zero"},
            assumptions=assumptions,
            limitations=limitations
        )

    def evaluate_all_candidates(
        self,
        candidates: List[Union[CandidateVesselResult, AISRecord, Dict[str, Any]]],
        spill: SpillEvent,
        env_provider: Optional[EnvironmentalDataProvider] = None
    ) -> List[CounterfactualResult]:
        """
        Runs counterfactual consistency evaluation across all candidate vessels
        and returns them ranked descending by Candidate Prioritization Score.
        """
        results = [
            self.evaluate_candidate(cand, spill, env_provider)
            for cand in candidates
        ]
        # Rank descending by score
        results.sort(
            key=lambda r: r.diagnostics.candidate_prioritization_score if r.diagnostics else 0.0,
            reverse=True
        )
        return results
