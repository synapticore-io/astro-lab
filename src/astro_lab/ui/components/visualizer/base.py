"""
Base Visualizer
===============

Abstract base class for all visualization backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import marimo as mo
import numpy as np


class BaseVisualizer(ABC):
    """Abstract base class for visualization backends."""

    def __init__(self):
        self.style_presets = {
            "cosmic_web": {
                "node_size": 2.0,
                "link_opacity": 0.5,
                "node_opacity": 0.8,
                "color_scheme": "survey",
                "show_grid": True,
            },
            "clusters": {
                "node_size": 3.0,
                "link_opacity": 0.3,
                "node_opacity": 0.9,
                "color_scheme": "cluster",
                "show_grid": False,
            },
            "filaments": {
                "node_size": 1.5,
                "link_opacity": 0.8,
                "node_opacity": 0.6,
                "color_scheme": "density",
                "show_grid": True,
            },
            "exploration": {
                "node_size": 2.5,
                "link_opacity": 0.4,
                "node_opacity": 0.7,
                "color_scheme": "magnitude",
                "show_grid": True,
            },
        }

    @abstractmethod
    def create_visualization(
        self, coords: np.ndarray, metadata: Dict[str, Any], params: Dict[str, Any]
    ) -> mo.Html:
        """Create visualization from coordinates and metadata.

        Args:
            coords: 3D coordinates array
            metadata: Additional data (magnitudes, clusters, etc.)
            params: Visualization parameters

        Returns:
            Marimo HTML element
        """

    def extract_coordinates(
        self, data: Any
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Extract coordinates and metadata from various data formats."""
        metadata = {}

        # Handle TensorDict
        if hasattr(data, "__class__") and "TensorDict" in data.__class__.__name__:
            if "coordinates" in data:
                coords = data["coordinates"].numpy()
            elif "pos" in data:
                coords = data["pos"].numpy()
            else:
                return None, metadata

            # Extract additional fields
            for key in ["magnitude", "velocity", "cluster_labels", "photometry"]:
                if key in data:
                    metadata[key] = data[key].numpy()

        # Handle Polars/Pandas DataFrame
        elif hasattr(data, "columns"):
            coord_cols = []
            if "x" in data.columns and "y" in data.columns and "z" in data.columns:
                coord_cols = ["x", "y", "z"]
            elif "ra" in data.columns and "dec" in data.columns:
                coord_cols = ["ra", "dec"]
                if "distance" in data.columns:
                    coord_cols.append("distance")

            if coord_cols:
                if hasattr(data, "to_numpy"):  # Polars
                    coords = data.select(coord_cols).to_numpy()
                else:  # Pandas
                    coords = data[coord_cols].to_numpy()

                # Extract metadata columns
                for col in ["magnitude", "velocity", "cluster_id", "g_mag", "bp_rp"]:
                    if col in data.columns:
                        if hasattr(data, "to_numpy"):
                            metadata[col] = data.select([col]).to_numpy().flatten()
                        else:
                            metadata[col] = data[col].to_numpy()
            else:
                return None, metadata

        # Handle numpy array
        elif isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[1] >= 2:
                coords = data[:, :3] if data.shape[1] >= 3 else data
            else:
                return None, metadata

        # Handle torch tensor
        elif hasattr(data, "numpy"):
            coords = data.numpy()
            if coords.ndim == 2 and coords.shape[1] >= 2:
                coords = coords[:, :3] if coords.shape[1] >= 3 else coords
            else:
                return None, metadata

        # Handle dict
        elif isinstance(data, dict):
            if "coordinates" in data:
                coords = np.array(data["coordinates"])
            elif "positions" in data:
                coords = np.array(data["positions"])
            elif "x" in data and "y" in data:
                x, y = np.array(data["x"]), np.array(data["y"])
                z = np.array(data.get("z", np.zeros_like(x)))
                coords = np.column_stack([x, y, z])
            else:
                return None, metadata

            # Copy other fields to metadata
            for key, value in data.items():
                if key not in ["coordinates", "positions", "x", "y", "z"]:
                    metadata[key] = (
                        np.array(value) if not isinstance(value, np.ndarray) else value
                    )
        else:
            return None, metadata

        # Ensure 3D coordinates
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        return coords, metadata

    def get_color_for_scheme(self, scheme: str) -> Tuple[float, float, float]:
        """Get default color for color scheme."""
        colors = {
            "survey": (0.8, 0.6, 0.1),  # Gold
            "cluster": (0.1, 0.5, 0.8),  # Blue
            "density": (0.8, 0.2, 0.2),  # Red
            "magnitude": (0.2, 0.8, 0.2),  # Green
        }
        return colors.get(scheme, (0.5, 0.5, 0.5))
