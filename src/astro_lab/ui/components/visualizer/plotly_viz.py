"""
Plotly Visualizer
=================

Interactive web-based visualization using Plotly.
"""

from typing import Any, Dict

import marimo as mo
import numpy as np

from astro_lab.widgets.plotly import (
    create_3d_analysis_plot,
    create_3d_scatter_plot,
)

from .base import BaseVisualizer


class PlotlyVisualizer(BaseVisualizer):
    """Plotly-based visualizer for interactive web plots."""

    def create_visualization(
        self, coords: np.ndarray, metadata: Dict[str, Any], params: Dict[str, Any]
    ) -> mo.Html:
        """Create Plotly visualization."""
        # Prepare color data
        color_data = None
        color_label = "Value"

        if params.get("color_scheme") == "cluster" and "cluster_labels" in metadata:
            color_data = metadata["cluster_labels"]
            color_label = "Cluster"
        elif params.get("color_scheme") == "magnitude" and "magnitude" in metadata:
            color_data = metadata["magnitude"]
            color_label = "Magnitude"
        elif params.get("color_scheme") == "density" and "density" in metadata:
            color_data = metadata["density"]
            color_label = "Density"

        # Create 3D scatter plot
        if "analysis_type" in metadata:
            return create_3d_analysis_plot(
                coords,
                color_data=color_data,
                color_label=color_label,
                size=params.get("node_size", 2.0),
                opacity=params.get("node_opacity", 0.8),
                title=f"{metadata['analysis_type']} Analysis",
                show_grid=params.get("show_grid", True),
            )
        else:
            return create_3d_scatter_plot(
                coords,
                color=color_data,
                size=params.get("node_size", 2.0),
                opacity=params.get("node_opacity", 0.8),
                title="3D Visualization",
            )
