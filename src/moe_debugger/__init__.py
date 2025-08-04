"""Explainable-MoE-Debugger: Chrome DevTools-style debugging for Mixture of Experts models.

This package provides real-time visualization and analysis tools for understanding
expert routing, load balancing, and token attribution in MoE architectures.
"""

from .__about__ import __version__

# Conditional imports to handle missing dependencies gracefully
__all__ = ["__version__"]

try:
    from .debugger import MoEDebugger
    __all__.append("MoEDebugger")
except ImportError:
    pass

try:
    from .analyzer import MoEAnalyzer  
    __all__.append("MoEAnalyzer")
except ImportError:
    pass

try:
    from .profiler import MoEProfiler
    __all__.append("MoEProfiler")
except ImportError:
    pass

try:
    from .server import DebugServer
    __all__.append("DebugServer")
except ImportError:
    pass