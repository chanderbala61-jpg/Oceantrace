"""
Environmental Data Providers and Vector Abstractions
----------------------------------------------------
Defines interface contracts and mock/synthetic providers for wind and ocean currents.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import math
from src.hindcasting.schema import EnvironmentalVector


class EnvironmentalDataProvider(ABC):
    """
    Abstract interface for spatial-temporal wind and ocean current providers.
    """
    @abstractmethod
    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Optional[EnvironmentalVector]:
        """Returns wind vector (u_eastward, v_northward) in m/s at the given location and time."""
        pass

    @abstractmethod
    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Optional[EnvironmentalVector]:
        """Returns surface ocean current vector (u, v) in m/s at the given location and time."""
        pass


def meteorological_to_vector(speed_ms: float, direction_from_deg: float, source: str = "custom") -> EnvironmentalVector:
    """
    Converts standard meteorological wind (speed in m/s, direction FROM which wind blows in degrees)
    into oceanographic vector components (u, v) representing the direction TOWARDS which the air moves.
    
    Formula:
      u = - speed * sin(radians(direction_from))
      v = - speed * cos(radians(direction_from))
    """
    if speed_ms < 0.0:
        raise ValueError(f"Speed must be non-negative, got {speed_ms}")
        
    rad = math.radians(direction_from_deg % 360.0)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    return EnvironmentalVector(u=u, v=v, source=source)


class ConstantEnvironmentalProvider(EnvironmentalDataProvider):
    """
    Deterministic provider returning constant or synthetic wind/current vectors.
    Useful for testing and controlled baselines.
    """
    def __init__(
        self,
        wind: Optional[EnvironmentalVector] = None,
        current: Optional[EnvironmentalVector] = None
    ):
        self.wind = wind
        self.current = current

    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Optional[EnvironmentalVector]:
        return self.wind

    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Optional[EnvironmentalVector]:
        return self.current
