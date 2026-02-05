"""
Enhanced Astronomical Tensor Bridge
==================================

Advanced bridge between AstroLab TensorDicts and visualization backends.
This module coordinates tensor conversion and visualization routing.
"""

import logging
from contextlib import contextmanager
from typing import Any

from .tensor_converters import converter

logger = logging.getLogger(__name__)


class AstronomicalTensorBridge:
    """
    Enhanced bridge that routes to the main widgets API with advanced features.

    This bridge provides:
    - Easy conversion between tensor formats and visualization backends
    - Automatic unit conversion for astronomical data
    - Memory-efficient data transfer
    - Support for all TensorDict types
    - Robust input validation and normalization

    Examples:
        >>> bridge = AstronomicalTensorBridge()
        >>>
        >>> # Convert SpatialTensorDict to visualization
        >>> viz = bridge.to_visualization(spatial_tensor, backend="pyvista")
        >>>
        >>> # Convert with unit conversion
        >>> viz = bridge.to_visualization(spatial_tensor, backend="cosmograph",
        ...                              convert_units=True, target_unit="Mpc")
    """

    def __init__(self):
        """Initialize the tensor bridge."""
        self.supported_backends = [
            "pyvista",
            "open3d",
            "blender",
            "plotly",
            "cosmograph",
        ]
        self._conversion_cache = {}
        self.converter = converter

    def to_visualization(self, tensordict: Any, backend: str = "auto", **kwargs) -> Any:
        """
        Route to main visualization API with enhanced features.

        Args:
            tensordict: Any TensorDict or tensor data
            backend: Target backend ("auto", "pyvista", "open3d", etc.)
            **kwargs: Additional parameters including:
                - convert_units: Whether to convert units (default: False)
                - target_unit: Target unit for conversion (default: keep original)
                - cache_result: Whether to cache conversion (default: False)

        Returns:
            Visualization object for the specified backend
        """
        from .. import create_visualization

        # Handle unit conversion if requested
        if kwargs.get("convert_units", False) and hasattr(tensordict, "coordinates"):
            target_unit = kwargs.pop("target_unit", "pc")
            tensordict = self.converter.convert_tensordict_units(
                tensordict, target_unit
            )

        # Auto-select backend if needed
        if backend == "auto":
            backend = self._auto_select_backend(tensordict, **kwargs)

        return create_visualization(tensordict, backend=backend, **kwargs)

    def _auto_select_backend(self, tensordict: Any, **kwargs) -> str:
        """Automatically select best backend based on data type and requirements."""
        # Check for specific requirements
        if kwargs.get("photorealistic", False):
            return "blender"
        elif kwargs.get("interactive", True):
            # Check data size
            features = self.converter.extract_features(tensordict)
            coords = features.get("coordinates")
            if coords is not None and len(coords) > 100000:
                return "open3d"  # Better for large point clouds
            else:
                return "pyvista"  # Good general purpose
        elif kwargs.get("web_export", False):
            return "plotly"
        else:
            return "pyvista"  # Default


@contextmanager
def tensor_bridge_context():
    """
    Context manager for tensor bridge operations.

    Examples:
        >>> with tensor_bridge_context() as bridge:
        ...     viz = bridge.to_visualization(spatial_tensor)
    """
    bridge = AstronomicalTensorBridge()
    try:
        yield bridge
    finally:
        # Clear any cached conversions
        bridge._conversion_cache.clear()


# Legacy compatibility
AstronomicalTensorZeroCopyBridge = AstronomicalTensorBridge

__all__ = [
    # Main classes
    "AstronomicalTensorBridge",
    # Context manager
    "tensor_bridge_context",
    # Legacy compatibility
    "AstronomicalTensorZeroCopyBridge",
]
