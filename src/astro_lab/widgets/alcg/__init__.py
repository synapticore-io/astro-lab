"""
AstroLab Cosmograph Integration (ALCG)
=====================================

Cosmograph integration for AstroLab with support for both TensorDict
and non-TensorDict data sources.

Based on:
- @cosmograph/cosmograph JavaScript library
- py_cosmograph Python bindings
- cosmograph_widget for Jupyter integration
"""

# Core bridge classes
from .bridge import (
    CosmographBridge,
    CosmographConfig,
    CosmographLinkData,
    CosmographNodeData,
)

# Convenience functions
from .convenience import (  # TensorDict-based functions; Non-TensorDict functions
    create_cosmic_web_cosmograph,
    create_cosmograph_from_coordinates,
    create_cosmograph_from_dataframe,
    create_cosmograph_from_tensordict,
    create_cosmograph_visualization,
    create_multimodal_cosmograph,
    visualize_analysis_results,
    visualize_spatial_tensordict,
)

__all__ = [
    # Core classes
    "CosmographBridge",
    "CosmographConfig",
    "CosmographNodeData",
    "CosmographLinkData",
    # TensorDict convenience functions
    "create_cosmograph_from_tensordict",
    "visualize_spatial_tensordict",
    "visualize_analysis_results",
    "create_cosmic_web_cosmograph",
    "create_multimodal_cosmograph",
    # Non-TensorDict convenience functions
    "create_cosmograph_from_coordinates",
    "create_cosmograph_from_dataframe",
    "create_cosmograph_visualization",
]
