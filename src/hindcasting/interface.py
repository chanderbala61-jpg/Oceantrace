"""
Abstract Hindcasting Model Interface
------------------------------------
Defines the standard execution signature for backward oil spill hindcasting engines.
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.ais.schema import SpillEvent
from src.hindcasting.schema import HindcastResult
from src.hindcasting.environmental import EnvironmentalDataProvider


class HindcastingModelInterface(ABC):
    """
    Contract for all backward oil spill transport and origin estimation models.
    """
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the hindcasting model."""
        pass

    @property
    @abstractmethod
    def model_level(self) -> int:
        """Hierarchy level (0: Direct Baseline, 1: Simple Wind, 2: Wind+Current, etc.)."""
        pass

    @abstractmethod
    def estimate_origin(
        self,
        spill: SpillEvent,
        drift_hours: float = 12.0,
        env_provider: Optional[EnvironmentalDataProvider] = None
    ) -> HindcastResult:
        """
        Estimates the historical release location and time window by backward simulation.
        
        Args:
            spill: The detected SpillEvent with observed centroid and observation time.
            drift_hours: Maximum lookback simulation duration in hours.
            env_provider: Optional environmental data provider supplying wind/current vectors.
            
        Returns:
            HindcastResult containing estimated origin, release window, trajectory, and uncertainty.
        """
        pass
