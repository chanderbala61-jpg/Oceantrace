"""
OceanTrace Oil Spill Hindcasting Package Entry Point
----------------------------------------------------
Exports core schemas, baseline models, and pipeline bridge helpers.
"""

from src.ais.schema import SpillEvent
from src.hindcasting.schema import HindcastResult, TrajectoryPoint, EnvironmentalVector
from src.hindcasting.interface import HindcastingModelInterface
from src.hindcasting.environmental import (
    EnvironmentalDataProvider,
    ConstantEnvironmentalProvider,
    meteorological_to_vector
)
from src.hindcasting.coordinates import displace_lat_lon, haversine_km
from src.hindcasting.models.direct_baseline import DirectObservationBaseline
from src.hindcasting.models.wind_drift import SimpleWindDriftModel
from src.hindcasting.visualization import export_hindcast_to_geojson, plot_hindcast_trajectory


def update_spill_event_from_hindcast(spill: SpillEvent, result: HindcastResult) -> SpillEvent:
    """
    Seamlessly updates a SpillEvent with the output of a HindcastResult,
    bridging the Hindcasting Engine directly to the AIS Correlation Engine.
    """
    spill.estimated_origin_lat = result.estimated_origin_lat
    spill.estimated_origin_lon = result.estimated_origin_lon
    spill.origin_time_start = result.origin_time_start
    spill.origin_time_end = result.origin_time_end
    spill.origin_uncertainty_km = result.origin_uncertainty_km
    return spill
