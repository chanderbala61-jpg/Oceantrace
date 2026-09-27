"""
OceanTrace Hindcasting Engine Interface
---------------------------------------
Abstract interface and contract for future ocean surface drift & wind-driven
hindcasting engines (e.g. OpenDrift, GNOME, HYCOM / Copernicus ocean currents).

IMPORTANT:
Does not fabricate ocean currents. If no validated oceanographic model is run,
preserves the detected observed centroid as the baseline anchor.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from src.ais.schema import SpillEvent


class HindcastingModelInterface(ABC):
    """
    Interface for backwards trajectory oil-spill hindcasting engines.
    """
    @abstractmethod
    def estimate_origin(
        self,
        spill: SpillEvent,
        drift_hours: float = 12.0
    ) -> SpillEvent:
        """
        Estimates the historical release location and time window by reverse-simulating
        wind and current drift.
        """
        pass


class DirectObservationBaseline(HindcastingModelInterface):
    """
    Default un-hindcasted baseline: assumes estimated origin equals the observed centroid,
    with an origin time window spanning up to 24 hours prior to the SAR acquisition.
    """
    def estimate_origin(
        self,
        spill: SpillEvent,
        drift_hours: float = 12.0
    ) -> SpillEvent:
        # Create a copy or updated instance
        spill.estimated_origin_lat = spill.centroid_lat
        spill.estimated_origin_lon = spill.centroid_lon
        spill.origin_time_end = spill.observation_time
        spill.origin_time_start = spill.observation_time - timedelta(hours=drift_hours)
        spill.origin_uncertainty_km = max(spill.origin_uncertainty_km, 15.0)
        return spill
