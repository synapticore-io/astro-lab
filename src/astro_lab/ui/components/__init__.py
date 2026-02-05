"""
UI Components for AstroLab
=========================

Simplified UI components using real AstroLab functionality.
"""

from .analyzer import create_analyzer, run_cosmic_web_analysis

# Core components that actually work
from .data_loader import create_data_loader
from .state import (
    create_state,
    get_analysis_config,
    get_data_config,
    get_visualization_config,
)
from .system_info import get_system_info
from .viz import create_cosmic_web_viz, create_plotly_viz, create_visualizer

# Only export what actually works
__all__ = [
    "create_data_loader",
    "create_analyzer",
    "run_cosmic_web_analysis",
    "create_visualizer",
    "create_plotly_viz",
    "create_cosmic_web_viz",
    "create_state",
    "get_data_config",
    "get_analysis_config",
    "get_visualization_config",
    "get_system_info",
]
