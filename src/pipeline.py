"""
OceanTrace End-to-End Investigation Pipeline
============================================
Unifies the core operational workflow:
  DETECT -> CHARACTERIZE -> TRACE -> CORRELATE -> INVESTIGATE

Stage Sequence:
  1. SAR Input & Preprocessing: Dual-polarization Sentinel-1 SAR (VH + VV).
  2. Detection / Segmentation: SegFormer-B0 inference or verified companion mask.
  3. Look-Alike & Ambiguity Assessment: Damping contrast, edge sharpness, aspect ratio, wind regime.
  4. Spill Characterization: Connected component extraction, centroid, area (km²), WGS-84 boundary.
  5. Environmental Drift Hindcasting: Backward trajectory, origin zone, source time window, uncertainty.
  6. AIS Candidate Vessel Correlation: Trajectory matching, closest approach, temporal offset, compatibility.
  7. Evidence Compilation & Output: Transparent audit trail with strict scientific integrity.

Scientific Integrity Standards:
  - Strict preservation of real data (no fabricated samples, tracks, or wind vectors).
  - Test set quarantined (never used for tuning).
  - Vessel results classified as "Candidate Vessels" with compatibility evidence, NOT legal accusations.
  - Graceful degradation with explicit warnings when environmental or AIS data is absent.
"""

import os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
import numpy as np

# Core internal schemas and modules
from src.ais.schema import SpillEvent, CandidateVesselResult
from src.characterization import SpillCharacterizer, parse_sentinel1_timestamp
from src.look_alike import LookAlikeAnalyzer, LookAlikeAssessment
from src.hindcasting.schema import HindcastResult, EnvironmentalVector
from src.hindcasting.interface import HindcastingModelInterface
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.models.direct_baseline import DirectObservationBaseline
from src.hindcasting.environmental import EnvironmentalDataProvider, ConstantEnvironmentalProvider
from src.hindcasting import update_spill_event_from_hindcast
from src.ais import run_ais_correlation_pipeline


@dataclass
class InvestigationRecord:
    """
    Unified result object documenting all stages of an OceanTrace investigation.
    """
    scene_id: str
    investigation_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Stage 1 & 2: Detection & Mask
    has_detection: bool = False
    detection_method: str = "CompanionMask"
    predicted_mask: Optional[np.ndarray] = None
    probability_map: Optional[np.ndarray] = None
    ground_truth_mask: Optional[np.ndarray] = None
    inference_metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostic_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 3: Look-Alike & Radiometric Ambiguity
    look_alike_assessment: Optional[LookAlikeAssessment] = None
    
    # Stage 4: Spill Characterization
    spill: Optional[SpillEvent] = None
    
    # Stage 5: Hindcasting & Source Reconstruction
    hindcast_result: Optional[HindcastResult] = None
    drift_model_name: Optional[str] = None
    
    # Stage 6: AIS Candidate Correlation
    ais_available: bool = False
    total_vessels_checked: int = 0
    candidate_vessels: List[CandidateVesselResult] = field(default_factory=list)
    ais_load_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 7: Counterfactual Vessel Testing (Experimental Add-On)
    counterfactual_results: List[Any] = field(default_factory=list)
    
    # Warnings, Assumptions, and Missing Data Tracking
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    missing_data_flags: Dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        """Produces a concise multi-line diagnostic summary."""
        lines = [
            f"=== OCEANTRACE INVESTIGATION: {self.scene_id} ===",
            f"Investigation Date (UTC): {self.investigation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Detection Status: {'Spill Detected' if self.has_detection else 'No Spill Detected'} (via {self.detection_method})",
        ]
        
        if self.look_alike_assessment:
            la = self.look_alike_assessment
            lines.append(f"Look-Alike Status: {la.classification} (Ambiguity Score: {la.ambiguity_score:.2f})")
            if la.is_look_alike_suspect:
                lines.append(f"  [!] Ambiguity Flags: {', '.join(la.flags)}")
        
        if self.spill:
            s = self.spill
            lines.append(f"Spill Location (Observed): {s.centroid_lat:.4f}°N, {s.centroid_lon:.4f}°E")
            lines.append(f"Spill Surface Area: {s.area_km2:.3f} km²")
            lines.append(f"Observation Time: {s.observation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
        if self.hindcast_result:
            h = self.hindcast_result
            lines.append(f"Estimated Origin: {h.estimated_origin_lat:.4f}°N, {h.estimated_origin_lon:.4f}°E (±{h.origin_uncertainty_km:.1f} km)")
            lines.append(f"Release Window: {h.origin_time_start.strftime('%H:%M')} - {h.origin_time_end.strftime('%H:%M UTC')}")
            lines.append(f"Simulated Drift Duration: {h.drift_duration_hours:.1f} hours via {h.model_name}")
            
        if self.ais_available:
            lines.append(f"AIS Correlation: Checked {self.total_vessels_checked} vessels, found {len(self.candidate_vessels)} candidates within search radius")
            for cand in self.candidate_vessels[:3]:
                m = cand.measurements
                v_name = m.get("ship_name") or f"MMSI {cand.mmsi}"
                d_km = m.get("min_distance_km", 0.0)
                dt_h = m.get("time_offset_hours", 0.0)
                lines.append(f"  - Rank #{cand.rank}: {v_name} | {cand.classification} (Score: {cand.overall_score:.2f}, Closest Approach: {d_km:.1f} km, dt: {dt_h:+.1f}h)")
        else:
            lines.append("AIS Correlation: UNAVAILABLE (No verified AIS transponder data provided)")

        if self.counterfactual_results:
            lines.append(f"Counterfactual Vessel Tests ({len(self.counterfactual_results)} candidates evaluated):")
            for cf in self.counterfactual_results[:3]:
                d = getattr(cf, "diagnostics", None)
                if d:
                    lines.append(
                        f"  - Candidate {cf.vessel_name} (MMSI: {cf.candidate_mmsi}): "
                        f"Score={d.candidate_prioritization_score:.3f} ({d.classification}), "
                        f"Endpoint Separation={d.centroid_distance_km:.2f} km"
                    )

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
                
        lines.append("==================================================")
        return "\n".join(lines)


def compute_pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pixel_spacing_m: float = 10.0
) -> Dict[str, Any]:
    """
    Computes rigorous pixel-level confusion matrix metrics.
    """
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

    union = tp + fp + fn
    diag_iou = float(tp / (union + 1e-7)) if union > 0 else (1.0 if fp == 0 and fn == 0 else 0.0)
    diag_dice = float(2 * tp / (2 * tp + fp + fn + 1e-7)) if (2 * tp + fp + fn) > 0 else (1.0 if fp == 0 and fn == 0 else 0.0)
    precision = float(tp / (tp + fp + 1e-7)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn + 1e-7)) if (tp + fn) > 0 else 0.0
    pixel_acc = float((tp + tn) / (tp + tn + fp + fn + 1e-7))

    px_area_km2 = (pixel_spacing_m * pixel_spacing_m) / 1e6
    fp_area_km2 = float(fp * px_area_km2)
    fn_area_km2 = float(fn * px_area_km2)
    fp_fraction = float(fp / (tp + fp + 1e-7)) if (tp + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "precision": precision,
        "recall": recall,
        "iou": diag_iou,
        "dice": diag_dice,
        "diagnostic_iou": diag_iou,
        "diagnostic_dice": diag_dice,
        "pixel_accuracy": pixel_acc,
        "fp_area_km2": fp_area_km2,
        "fn_area_km2": fn_area_km2,
        "fp_fraction": fp_fraction,
    }


def classify_scene_detection(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies a scene based on pixel-level confusion matrix metrics into
    defensible scientific categories.
    """
    tp = metrics.get("tp", metrics.get("tp_pixels", 0))
    fp = metrics.get("fp", metrics.get("fp_pixels", 0))
    fn = metrics.get("fn", metrics.get("fn_pixels", 0))
    diag_iou = metrics.get("iou", metrics.get("diagnostic_iou", 0.0))

    pred_active = tp + fp
    gt_active = tp + fn

    if pred_active == 0 and gt_active == 0:
        return {
            "category": "TRUE_NEGATIVE",
            "label": "True Negative Scene (Zero Slicks Observed)",
            "explanation": "Neither ground truth nor model predicted oil."
        }
    elif pred_active == 0 and gt_active > 0:
        return {
            "category": "FALSE_NEGATIVE",
            "label": "False Negative Scene (Undetected Ground-Truth Slick)",
            "explanation": "Ground truth contains oil but model missed it."
        }
    elif pred_active > 0 and gt_active == 0:
        return {
            "category": "PURE_FALSE_POSITIVE",
            "label": "Pure False Positive Scene (Zero Ground-Truth Overlap)",
            "explanation": "Model predicted oil where ground truth has none."
        }
    elif diag_iou < 0.50:
        return {
            "category": "PARTIAL_DETECTION",
            "label": "Partial Detection / Boundary Disparity",
            "explanation": f"Model detected oil with partial ground-truth overlap (IoU = {diag_iou:.2f})."
        }
    else:
        return {
            "category": "STRONG_DETECTION",
            "label": "Validated Detection with Ground-Truth Overlap",
            "explanation": f"High-confidence detection with substantial ground-truth overlap (IoU = {diag_iou:.2f})."
        }


class OceanTracePipeline:
    """
    Autonomous End-to-End Pipeline Orchestrator.
    """

    def __init__(
        self,
        characterizer: Optional[SpillCharacterizer] = None,
        look_alike_analyzer: Optional[LookAlikeAnalyzer] = None,
        drift_model: Optional[HindcastingModelInterface] = None,
        model: Optional[Any] = None,
        default_drift_hours: float = 12.0,
        ais_search_radius_km: float = 50.0,
        ais_lookback_hours: float = 24.0,
        ais_lookforward_hours: float = 6.0,
    ):
        self.characterizer = characterizer or SpillCharacterizer()
        self.look_alike_analyzer = look_alike_analyzer or LookAlikeAnalyzer()
        self.drift_model = drift_model or SimpleWindDriftModel(windage_factor=0.03)
        self.model = model
        self.default_drift_hours = float(default_drift_hours)
        self.ais_search_radius_km = float(ais_search_radius_km)
        self.ais_lookback_hours = float(ais_lookback_hours)
        self.ais_lookforward_hours = float(ais_lookforward_hours)

    def execute(
        self,
        scene_id: str,
        sar_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        observation_time: Optional[Union[str, datetime]] = None,
        origin_lat_hint: Optional[float] = None,
        origin_lon_hint: Optional[float] = None,
        transform: Optional[Any] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crs: Optional[Any] = None,
        env_provider: Optional[EnvironmentalDataProvider] = None,
        ais_csv_path: Optional[str] = None,
        wind_speed_ms: Optional[float] = None,
        drift_hours: Optional[float] = None,
        model: Optional[Any] = None,
        run_inference: bool = False,
        inference_patch_size: int = 256,
        inference_stride: int = 256,
        run_counterfactual: bool = False,
    ) -> InvestigationRecord:
        """
        Executes the complete investigation pipeline on a single SAR observation.
        """
        record = InvestigationRecord(scene_id=scene_id)
        drift_hrs = float(drift_hours or self.default_drift_hours)
        eval_model = model or self.model

        active_mask = None

        # ----------------------------------------------------
        # Stage 1 & 2: Detection / Inference & Mask Selection
        # ----------------------------------------------------
        if (run_inference or (mask is None and eval_model is not None)) and sar_image is not None:
            try:
                from src.inference import predict_scene_tiled
                pred_res = predict_scene_tiled(
                    eval_model,
                    sar_image,
                    patch_size=inference_patch_size,
                    stride=inference_stride,
                )
                record.predicted_mask = pred_res["binary_mask"]
                record.probability_map = pred_res["probability_map"]
                record.inference_metadata = {
                    k: v for k, v in pred_res.items()
                    if k not in ("probability_map", "binary_mask")
                }
                record.detection_method = "SegFormer-B0 Inference (Frozen best.pth)"
                active_mask = record.predicted_mask

                # If a companion mask is also supplied, preserve it as ground truth
                if mask is not None:
                    record.ground_truth_mask = mask
                    metrics = compute_pixel_metrics(active_mask, mask, pixel_spacing_m=10.0)
                    scene_diag = classify_scene_detection(metrics)
                    record.diagnostic_metrics = {
                        **metrics,
                        "scene_classification": scene_diag["label"],
                        "scene_category": scene_diag["category"],
                        "diagnostic_explanation": scene_diag["explanation"],
                        "comparison_note": "Diagnostic scene-level comparison. Does not alter locked test results.",
                    }
            except Exception as e:
                record.warnings.append(f"SegFormer inference failed: {e}. Attempting fallback to companion mask.")
                if mask is not None:
                    active_mask = mask
                    record.detection_method = "VerifiedCompanionMask"
        elif mask is not None:
            active_mask = mask
            record.detection_method = "VerifiedCompanionMask"

        if active_mask is None or np.sum(active_mask > 0.5) == 0:
            record.has_detection = False
            record.warnings.append("No oil spill pixels detected in the input segmentation mask.")
            return record

        record.has_detection = True

        # ----------------------------------------------------
        # Stage 3: Look-Alike & Ambiguity Assessment
        # ----------------------------------------------------
        if sar_image is not None:
            try:
                look_alike_res = self.look_alike_analyzer.assess(
                    spill_id=f"spill_{scene_id}",
                    mask=active_mask,
                    sar_image=sar_image,
                    wind_speed_ms=wind_speed_ms,
                )
                record.look_alike_assessment = look_alike_res
                if look_alike_res.warnings:
                    record.warnings.extend(look_alike_res.warnings)
            except Exception as e:
                record.warnings.append(f"Look-alike assessment encountered an error: {e}")
        else:
            record.missing_data_flags["sar_radiometry"] = True
            record.warnings.append("SAR radiometry not provided; skipping detailed damping contrast look-alike analysis.")

        # ----------------------------------------------------
        # Stage 4: Spill Characterization -> SpillEvent
        # ----------------------------------------------------
        obs_dt = parse_sentinel1_timestamp(observation_time or datetime.utcnow())
        spill = self.characterizer.characterize(
            mask=active_mask,
            spill_id=f"spill_{scene_id}",
            observation_time=obs_dt,
            transform=transform,
            bbox=bbox,
            origin_lat_hint=origin_lat_hint,
            origin_lon_hint=origin_lon_hint,
            crs=crs,
        )

        if spill is None:
            record.has_detection = False
            record.warnings.append(
                f"Detection filtered out: Slick area below minimum threshold ({self.characterizer.min_slick_pixels} px)."
            )
            return record

        record.spill = spill

        # ----------------------------------------------------
        # Stage 5: Backward Environmental Drift Hindcasting
        # ----------------------------------------------------
        try:
            hindcast_res = self.drift_model.estimate_origin(
                spill=spill,
                drift_hours=drift_hrs,
                env_provider=env_provider
            )
            record.hindcast_result = hindcast_res
            record.drift_model_name = self.drift_model.model_name
            record.assumptions.extend(hindcast_res.assumptions)
            record.warnings.extend(hindcast_res.warnings)

            # Update SpillEvent with hindcasted origin
            update_spill_event_from_hindcast(spill, hindcast_res)
        except Exception as e:
            record.warnings.append(f"Hindcasting failed with error: {e}. Falling back to observed centroid.")
            record.missing_data_flags["hindcasting_failure"] = True
            # Level 0 direct baseline fallback
            fallback_model = DirectObservationBaseline()
            hindcast_res = fallback_model.estimate_origin(spill, drift_hours=drift_hrs)
            record.hindcast_result = hindcast_res
            record.drift_model_name = fallback_model.model_name
            update_spill_event_from_hindcast(spill, hindcast_res)

        # ----------------------------------------------------
        # Stage 6: AIS Candidate Correlation
        # ----------------------------------------------------
        if ais_csv_path and os.path.isfile(ais_csv_path):
            try:
                ais_out = run_ais_correlation_pipeline(
                    spill=spill,
                    ais_csv_path=ais_csv_path,
                    lookback_hours=self.ais_lookback_hours,
                    lookforward_hours=self.ais_lookforward_hours,
                    search_radius_km=self.ais_search_radius_km,
                )
                record.ais_available = True
                record.total_vessels_checked = ais_out.get("total_vessels_in_time_window", 0)
                record.candidate_vessels = ais_out.get("ranked_candidates", [])
                record.ais_load_summary = ais_out.get("load_summary", {})
            except Exception as e:
                record.ais_available = False
                record.missing_data_flags["ais_correlation_error"] = True
                record.warnings.append(f"AIS correlation pipeline error: {e}")
        else:
            record.ais_available = False
            record.missing_data_flags["ais_data_missing"] = True
            record.warnings.append(
                "AIS data unavailable: No AIS broadcast repository provided for this geographic area and timeframe."
            )

        # ----------------------------------------------------
        # Stage 7: Counterfactual Vessel Testing (Experimental)
        # ----------------------------------------------------
        if run_counterfactual and record.candidate_vessels and record.spill:
            try:
                from src.counterfactual import CounterfactualAnalyzer
                cf_analyzer = CounterfactualAnalyzer()
                record.counterfactual_results = cf_analyzer.evaluate_all_candidates(
                    candidates=record.candidate_vessels,
                    spill=record.spill,
                    env_provider=env_provider
                )
            except Exception as e:
                record.warnings.append(f"Counterfactual vessel testing encountered an error: {e}")

        return record
