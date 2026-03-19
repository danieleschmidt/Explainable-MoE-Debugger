"""
Explainable-MoE-Debugger
========================
Visualization and explainability tools for Mixture-of-Experts routing decisions.

Core components:
    MoELayer             - MoE layer with built-in routing recorder
    RoutingExplainer     - Gradient-based attribution per expert
    ExpertActivationMap  - Feature activation profiles per expert
    DeadExpertDetector   - Detects underutilized / collapsed experts
    MoEDebugReport       - Structured diagnostic report
"""

from .layer import MoELayer
from .explainer import RoutingExplainer
from .activation_map import ExpertActivationMap
from .dead_expert import DeadExpertDetector
from .report import MoEDebugReport

__all__ = [
    "MoELayer",
    "RoutingExplainer",
    "ExpertActivationMap",
    "DeadExpertDetector",
    "MoEDebugReport",
]
