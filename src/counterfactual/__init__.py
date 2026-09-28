"""
OceanTrace Counterfactual Vessel Testing Package
================================================
Extends the AIS investigation pipeline with forward 4th-Order Runge-Kutta
drift simulations from candidate vessel positions to evaluate spatiotemporal
consistency with observed SAR slicks.

Key Components:
  - HypotheticalRelease: Container for candidate release events
  - ForwardSimulationConfig: Parameters for numerical integration
  - ForwardRK4DriftModel: Forward Lagrangian Runge-Kutta drift engine
  - ConsistencyDiagnostics: Quantifiable geometric and temporal metrics
  - CounterfactualResult: Full diagnostic result container
  - CounterfactualAnalyzer: Primary evaluation orchestrator
"""

from src.counterfactual.schema import (
    HypotheticalRelease,
    ForwardSimulationConfig,
    ConsistencyDiagnostics,
    CounterfactualResult,
)
from src.counterfactual.forward_rk4 import ForwardRK4DriftModel
from src.counterfactual.analyzer import CounterfactualAnalyzer

__all__ = [
    "HypotheticalRelease",
    "ForwardSimulationConfig",
    "ConsistencyDiagnostics",
    "CounterfactualResult",
    "ForwardRK4DriftModel",
    "CounterfactualAnalyzer",
]
