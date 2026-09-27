"""
OceanTrace SAR Look-Alike & Ambiguity Assessment Module
-------------------------------------------------------
Provides explainable, physics-grounded ambiguity assessment for SAR dark features.

SAR dark formations can originate from:
  - Mineral oil spills (authentic target)
  - Low-wind / calm sea surface areas (< 3 m/s)
  - Biogenic slicks / natural organic films
  - Vessel wakes / shear zones
  - Rain cells / atmospheric downdrafts
  - Sea-state variations / internal waves

Scientific Policy:
  Because the training dataset provides binary spill masks without dedicated
  fine-grained look-alike class labels, this module NEVER invents synthetic labels.
  Instead, it computes genuinely measurable radiometric contrast, morphological
  ratios, edge gradients, and environmental wind context to quantify ambiguity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2


@dataclass
class LookAlikeAssessment:
    """
    Explainable assessment of whether a detected SAR dark area exhibits
    characteristics of genuine mineral oil or potential look-alikes.
    """
    spill_id: str
    ambiguity_level: str  # 'LOW', 'MODERATE', 'HIGH'
    ambiguity_score: float  # [0.0 = high confidence spill, 1.0 = high look-alike ambiguity]
    
    # Radiometric contrast between slick and surrounding background sea (in dB)
    contrast_db: Optional[float] = None
    
    # Boundary gradient sharpness (higher = sharper boundary typical of oil surface tension)
    edge_gradient: Optional[float] = None
    
    # Morphological aspect ratio (major axis / minor axis length)
    aspect_ratio: Optional[float] = None
    
    # Compactness / circularity: 4 * pi * area / perimeter^2
    compactness: Optional[float] = None
    
    # Wind speed at observation time (if available)
    wind_speed_ms: Optional[float] = None
    
    # Explainable flags and warnings
    flags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_reliable_detection(self) -> bool:
        """True if detection does not have HIGH ambiguity."""
        return self.ambiguity_level != "HIGH"

    @property
    def is_look_alike_suspect(self) -> bool:
        """True if detection exhibits HIGH look-alike ambiguity."""
        return self.ambiguity_level == "HIGH"

    @property
    def damping_contrast_db(self) -> Optional[float]:
        """Convenience alias for contrast_db."""
        return self.contrast_db

    @property
    def edge_gradient_mean(self) -> Optional[float]:
        """Convenience alias for edge_gradient."""
        return self.edge_gradient

    @property
    def classification(self) -> str:
        """Human-readable ambiguity classification."""
        if self.ambiguity_level == "LOW":
            return "True Oil Spill Candidate"
        elif self.ambiguity_level == "MODERATE":
            return "Possible Oil Spill (Moderate Ambiguity)"
        else:
            return "Look-Alike Suspect (High Ambiguity)"


class LookAlikeAnalyzer:
    """
    Analyzes SAR image dual-polarization values and mask morphology
    to assess false-positive / look-alike ambiguity.
    """

    def __init__(
        self,
        min_contrast_db: float = 3.5,
        optimal_contrast_db: float = 7.0,
        low_wind_threshold_ms: float = 3.0,
        high_wind_threshold_ms: float = 12.0,
    ):
        self.min_contrast_db = float(min_contrast_db)
        self.optimal_contrast_db = float(optimal_contrast_db)
        self.low_wind_threshold_ms = float(low_wind_threshold_ms)
        self.high_wind_threshold_ms = float(high_wind_threshold_ms)

    def assess(
        self,
        spill_id: str,
        mask: np.ndarray,
        sar_image: Optional[np.ndarray] = None,
        wind_speed_ms: Optional[float] = None,
    ) -> LookAlikeAssessment:
        """
        Computes explainable ambiguity assessment for a detected mask.

        Args:
            spill_id: Identifier for the spill event.
            mask: 2D binary numpy array (H, W).
            sar_image: Optional (H, W) or (2, H, W) float32 SAR backscatter in dB.
            wind_speed_ms: Optional scalar wind speed (m/s) at acquisition time.

        Returns:
            LookAlikeAssessment object.
        """
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask_binary = (mask > 0.5).astype(np.uint8)

        flags: List[str] = []
        warnings: List[str] = []
        evidence: Dict[str, Any] = {}

        # 1. Morphological Features
        area_pixels = int(np.sum(mask_binary))
        if area_pixels == 0:
            return LookAlikeAssessment(
                spill_id=spill_id,
                ambiguity_level="HIGH",
                ambiguity_score=1.0,
                flags=["NO_ACTIVE_SLICK"],
                warnings=["Mask contains 0 slick pixels."],
            )

        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea) if contours else None

        aspect_ratio: Optional[float] = None
        compactness: Optional[float] = None

        if largest_contour is not None and len(largest_contour) >= 5:
            # Fit ellipse to determine aspect ratio
            ellipse = cv2.fitEllipse(largest_contour)
            (_, _), (axis1, axis2), _ = ellipse
            major_axis = max(axis1, axis2)
            minor_axis = max(min(axis1, axis2), 1e-3)
            aspect_ratio = float(major_axis / minor_axis)

            # Perimeter and compactness
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                compactness = float((4.0 * np.pi * area_pixels) / (perimeter * perimeter))
        elif largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(max(w, h) / max(min(w, h), 1))
            compactness = 0.5

        evidence["area_pixels"] = area_pixels
        evidence["aspect_ratio"] = round(aspect_ratio, 2) if aspect_ratio is not None else None
        evidence["compactness"] = round(compactness, 4) if compactness is not None else None

        # Check aspect ratio implications
        if aspect_ratio is not None and aspect_ratio > 10.0:
            flags.append("HIGH_ELONGATION_LINEAR_STRUCTURE")
            warnings.append("High elongation (>10.0): May represent a ship wake, internal wave train, or point-source trail.")
        elif aspect_ratio is not None and aspect_ratio < 1.3 and area_pixels > 50000:
            flags.append("BROAD_DIFFUSE_FORMATION")
            warnings.append("Broad, rounded formation: May represent calm low-wind water surface rather than trailing slick.")

        # 2. Radiometric Contrast (if SAR image provided)
        contrast_db: Optional[float] = None
        edge_gradient: Optional[float] = None

        if sar_image is not None:
            # Extract single channel (prefer VV if 2 channels [VH, VV], else 2D)
            if sar_image.ndim == 3 and sar_image.shape[0] == 2:
                # Channel 1 is VV, Channel 0 is VH
                backscatter = sar_image[1]
            elif sar_image.ndim == 3 and sar_image.shape[-1] == 2:
                backscatter = sar_image[..., 1]
            else:
                backscatter = sar_image.squeeze()

            # Create dilated background sea mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate(mask_binary, kernel, iterations=2)
            sea_background_mask = (dilated_mask == 1) & (mask_binary == 0)

            slick_vals = backscatter[mask_binary == 1]
            sea_vals = backscatter[sea_background_mask]

            if len(slick_vals) > 0 and len(sea_vals) > 0:
                slick_mean = float(np.nanmean(slick_vals))
                sea_mean = float(np.nanmean(sea_vals))
                # Contrast: sea is brighter than dark slick
                contrast_db = float(sea_mean - slick_mean)
                evidence["slick_mean_db"] = round(slick_mean, 2)
                evidence["sea_mean_db"] = round(sea_mean, 2)
                evidence["measured_contrast_db"] = round(contrast_db, 2)

                if contrast_db < self.min_contrast_db:
                    flags.append("LOW_RADIOMETRIC_CONTRAST")
                    warnings.append(
                        f"Low radiometric contrast ({contrast_db:.2f} dB < {self.min_contrast_db} dB): "
                        "Darkness relative to sea background is weak; could be natural biogenic slick or sea clutter."
                    )
                else:
                    evidence["contrast_status"] = "Damping consistent with mineral oil damping"

            # Compute edge gradient sharpness across boundary
            grad_x = cv2.Sobel(backscatter, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(backscatter, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            boundary_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            edge_vals = grad_mag[boundary_mask > 0]
            if len(edge_vals) > 0:
                edge_gradient = float(np.nanmean(edge_vals))
                evidence["boundary_gradient_mean"] = round(edge_gradient, 3)

        # 3. Environmental Wind Assessment
        if wind_speed_ms is not None:
            evidence["wind_speed_ms"] = float(wind_speed_ms)
            if wind_speed_ms < self.low_wind_threshold_ms:
                flags.append("LOW_WIND_AMBIGUITY_ZONE")
                warnings.append(
                    f"Low wind speed ({wind_speed_ms:.1f} m/s < {self.low_wind_threshold_ms} m/s): "
                    "SAR surface backscatter is naturally attenuated in calm water, creating dark look-alikes."
                )
            elif wind_speed_ms > self.high_wind_threshold_ms:
                flags.append("HIGH_WIND_DISPERSION_ZONE")
                warnings.append(
                    f"High wind speed ({wind_speed_ms:.1f} m/s > {self.high_wind_threshold_ms} m/s): "
                    "Wave breaking accelerates oil dissipation and reduces SAR detection contrast."
                )
            else:
                evidence["wind_condition"] = "Optimal wind regime (3-12 m/s) for SAR oil spill detection"

        # 4. Synthesize Ambiguity Score and Level
        # Baseline ambiguity is 0.15 (inherent uncertainty without chemical sampling)
        score = 0.15

        if "LOW_WIND_AMBIGUITY_ZONE" in flags:
            score += 0.40
        if "LOW_RADIOMETRIC_CONTRAST" in flags:
            score += 0.30
        if "HIGH_WIND_DISPERSION_ZONE" in flags:
            score += 0.15
        if "BROAD_DIFFUSE_FORMATION" in flags:
            score += 0.15

        # Reductions if clear oil indicators are present
        if contrast_db is not None and contrast_db >= self.optimal_contrast_db:
            score = max(0.05, score - 0.20)

        score = float(np.clip(score, 0.0, 1.0))

        if score < 0.35:
            ambiguity_level = "LOW"
        elif score < 0.65:
            ambiguity_level = "MODERATE"
        else:
            ambiguity_level = "HIGH"

        return LookAlikeAssessment(
            spill_id=str(spill_id),
            ambiguity_level=ambiguity_level,
            ambiguity_score=round(score, 3),
            contrast_db=round(contrast_db, 2) if contrast_db is not None else None,
            edge_gradient=round(edge_gradient, 3) if edge_gradient is not None else None,
            aspect_ratio=round(aspect_ratio, 2) if aspect_ratio is not None else None,
            compactness=round(compactness, 4) if compactness is not None else None,
            wind_speed_ms=float(wind_speed_ms) if wind_speed_ms is not None else None,
            flags=flags,
            evidence=evidence,
            warnings=warnings,
        )
