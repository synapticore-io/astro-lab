"""
Cosmograph Visualizer
====================

GPU-accelerated interactive visualization using Cosmograph.
"""

from typing import Any, Dict

import marimo as mo
import numpy as np

from astro_lab.widgets.alcg import (
    create_cosmic_web_cosmograph,
    create_cosmograph_visualization,
)

from .base import BaseVisualizer


class CosmographVisualizer(BaseVisualizer):
    """Cosmograph-based visualizer for large-scale astronomical data."""

    def create_visualization(
        self, coords: np.ndarray, metadata: Dict[str, Any], params: Dict[str, Any]
    ) -> mo.Html:
        """Create Cosmograph visualization."""
        # Check if we have cosmic web analysis results
        if any(
            key in metadata for key in ["cluster_labels", "filament_edges", "void_mask"]
        ):
            return create_cosmic_web_cosmograph(
                coords,
                analysis_results=metadata,
                node_size_scale=params.get("node_size", 2.0),
                link_opacity=params.get("link_opacity", 0.5),
                node_opacity=params.get("node_opacity", 0.8),
                show_clusters="cluster_labels" in metadata,
                show_filaments="filament_edges" in metadata,
                show_voids="void_mask" in metadata,
            )
        else:
            # Basic visualization
            return create_cosmograph_visualization(
                coords,
                node_size=params.get("node_size", 2.0),
                opacity=params.get("node_opacity", 0.8),
                survey=metadata.get("survey", "auto"),
                show_grid=params.get("show_grid", True),
                magnitudes=metadata.get("magnitude"),
            )
