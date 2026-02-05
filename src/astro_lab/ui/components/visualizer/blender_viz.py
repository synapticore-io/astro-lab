"""
Blender Visualizer
==================

High-quality 3D rendering using Blender.
"""

from typing import Any, Dict

import marimo as mo
import numpy as np

from astro_lab.widgets.albpy import (
    create_blender_scene,
    render_cosmic_web,
)

from .base import BaseVisualizer


class BlenderVisualizer(BaseVisualizer):
    """Blender-based visualizer for high-quality rendering."""

    def create_visualization(
        self, coords: np.ndarray, metadata: Dict[str, Any], params: Dict[str, Any]
    ) -> mo.Html:
        """Create Blender visualization."""
        # Check if we have cosmic web data
        if any(key in metadata for key in ["cluster_labels", "filament_edges"]):
            scene = render_cosmic_web(
                coords,
                clusters=metadata.get("cluster_labels"),
                filaments=metadata.get("filament_edges"),
                voids=metadata.get("void_mask"),
                point_size=params.get("node_size", 2.0),
                quality="high",
            )
        else:
            scene = create_blender_scene(
                coords,
                point_size=params.get("node_size", 2.0),
                point_color=self.get_color_for_scheme(
                    params.get("color_scheme", "survey")
                ),
                render_quality="high",
            )

        return scene
