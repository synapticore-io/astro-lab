"""Backend-specific conversion utilities.

This module provides convenience functions for converting data
to different visualization backend formats.
"""

from typing import Any, Dict, Union

import numpy as np

from ..enhanced.tensor_bridge import AstronomicalTensorBridge

# Initialize bridge
bridge = AstronomicalTensorBridge()


def to_pyvista(
    data: Union[Dict[str, Any], np.ndarray],
    convert_units: bool = False,
    target_unit: str = "pc",
    **kwargs,
) -> Any:
    """Convert data to PyVista visualization.

    Args:
        data: Input data (dict or array)
        convert_units: Whether to convert units
        target_unit: Target unit for coordinates
        **kwargs: PyVista-specific options

    Returns:
        PyVista object (mesh, plotter, etc.)
    """
    return bridge.to_visualization(
        data,
        backend="pyvista",
        convert_units=convert_units,
        target_unit=target_unit,
        **kwargs,
    )


def to_open3d(data: Union[Dict[str, Any], np.ndarray], **kwargs) -> Any:
    """Convert to Open3D visualization.

    Args:
        data: Input data
        **kwargs: Open3D-specific options

    Returns:
        Open3D geometry object
    """
    return bridge.to_visualization(data, backend="open3d", **kwargs)


def to_blender(data: Union[Dict[str, Any], np.ndarray], **kwargs) -> Any:
    """Convert to Blender objects.

    Args:
        data: Input data
        **kwargs: Blender-specific options

    Returns:
        Blender object(s)
    """
    return bridge.to_visualization(data, backend="blender", **kwargs)


def to_plotly(
    data: Union[Dict[str, Any], np.ndarray], plot_type: str = "scatter", **kwargs
) -> Any:
    """Convert to Plotly visualization.

    Args:
        data: Input data
        plot_type: Type of plot
        **kwargs: Plotly-specific options

    Returns:
        Plotly figure
    """
    return bridge.to_visualization(
        data, backend="plotly", plot_type=plot_type, **kwargs
    )


def to_cosmograph(
    data: Union[Dict[str, Any], np.ndarray],
    build_graph: bool = True,
    k_neighbors: int = 8,
    **kwargs,
) -> Any:
    """Convert to Cosmograph visualization.

    Args:
        data: Input data
        build_graph: Whether to build graph structure
        k_neighbors: Number of neighbors for graph construction
        **kwargs: Cosmograph-specific options

    Returns:
        Cosmograph-compatible data
    """
    return bridge.to_visualization(
        data,
        backend="cosmograph",
        build_graph=build_graph,
        k_neighbors=k_neighbors,
        **kwargs,
    )


# Legacy compatibility functions
def transfer_astronomical_tensor(
    tensor: Any, backend: str = "pyvista", **kwargs
) -> Any:
    """Legacy tensor transfer function for backward compatibility."""
    return bridge.to_visualization(tensor, backend=backend, **kwargs)


def astronomical_tensor_zero_copy_context():
    """Legacy context manager for backward compatibility."""
    from .tensor_bridge import tensor_bridge_context

    return tensor_bridge_context()


__all__ = [
    # Backend converters
    "to_pyvista",
    "to_open3d",
    "to_blender",
    "to_plotly",
    "to_cosmograph",
    # Legacy compatibility
    "transfer_astronomical_tensor",
    "astronomical_tensor_zero_copy_context",
    # Global instances
    "bridge",
]
