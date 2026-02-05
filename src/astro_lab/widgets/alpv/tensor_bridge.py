"""
PyVista Tensor Bridge for AstroLab - Astronomical Data Visualization
===================================================================

Bridge between AstroLab tensors and PyVista for 3D astronomical visualization.

Note: This module uses the Enhanced-API for all 3D visualization (see astro_lab.widgets.enhanced).
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import quantity_support

from astro_lab.widgets.enhanced import to_pyvista

# Enable quantity support
quantity_support()

logger = logging.getLogger(__name__)

# =========================================================================
# ASTRONOMICAL MEMORY MANAGEMENT
# =========================================================================


class AstronomicalPyVistaMeshManager:
    """Manages PyVista PolyData objects with astronomical context."""

    def __init__(self):
        self.active_meshes = set()
        self.astronomical_units = {}  # Track units for each mesh

    def register_mesh(self, mesh, unit: str = "pc"):
        """Register a mesh for cleanup with astronomical unit tracking."""
        self.active_meshes.add(mesh)
        self.astronomical_units[mesh] = unit
        logger.debug(f"Registered astronomical PyVista mesh with unit: {unit}")

    def cleanup_all(self):
        """Clean up all registered meshes."""
        for mesh in self.active_meshes:
            self.safe_polydata_del(mesh)
        self.active_meshes.clear()
        self.astronomical_units.clear()

    def safe_polydata_del(self, mesh):
        """Safely delete PyVista PolyData object."""
        try:
            if hasattr(mesh, "clear_data"):
                mesh.clear_data()
            if hasattr(mesh, "clear_points"):
                mesh.clear_points()
            if hasattr(mesh, "clear_cells"):
                mesh.clear_cells()
            del mesh
        except Exception as e:
            logger.warning(f"Error cleaning up astronomical PyVista PolyData: {e}")


# Global astronomical mesh manager
_astronomical_pyvista_mesh_manager = AstronomicalPyVistaMeshManager()


@dataclass
class AstronomicalPyVistaSyncConfig:
    """Configuration for PyVista synchronization with astronomical context."""

    sync_interval: float = 0.1  # seconds
    auto_sync: bool = True
    preserve_names: bool = True
    zero_copy: bool = True
    max_vertices: int = 1000000  # Performance limit
    coordinate_system: str = "icrs"  # Astronomical coordinate system
    distance_unit: str = "pc"  # Astronomical distance unit
    magnitude_unit: str = "mag"  # Astronomical magnitude unit


# =========================================================================
# Astronomical memory profiling and optimization utilities
# =========================================================================


@contextmanager
def astronomical_pyvista_zero_copy_context(
    description: str = "PyVista astronomical zero-copy operation",
):
    """
    Context manager for PyVista zero-copy operations with astronomical memory profiling.

    Args:
        description: Description for profiling logs

    Yields:
        dict: Memory statistics during operation
    """
    initial_stats = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_stats = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }

    try:
        yield initial_stats
    finally:
        # Cleanup and profiling
        if torch.cuda.is_available():
            final_stats = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
            }

            memory_diff = final_stats["allocated"] - initial_stats.get("allocated", 0)
            if memory_diff > 1024**2:  # More than 1MB increase
                logger.warning(
                    f"PyVista astronomical zero-copy {description}: Memory increase {memory_diff / 1024**2:.2f} MB"
                )
            else:
                logger.debug(
                    f"PyVista astronomical zero-copy {description}: Memory change {memory_diff / 1024**2:.2f} MB"
                )


def optimize_astronomical_tensor_layout(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimize tensor memory layout for PyVista astronomical zero-copy operations.

    Args:
        tensor: Input astronomical tensor

    Returns:
        Optimized tensor (contiguous and detached)
    """
    # Always detach to prevent autograd issues
    optimized = tensor.detach()

    # Ensure contiguous layout for zero-copy
    if not optimized.is_contiguous():
        optimized = optimized.contiguous()

    return optimized


def get_astronomical_tensor_memory_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get comprehensive memory information for an astronomical tensor.

    Args:
        tensor: Input astronomical tensor

    Returns:
        Dictionary with memory statistics
    """
    info = {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "storage_size": tensor.untyped_storage().size()
        if hasattr(tensor, "untyped_storage")
        else tensor.storage().size(),
        "data_ptr": tensor.data_ptr(),
        "is_contiguous": tensor.is_contiguous(),
        "requires_grad": tensor.requires_grad,
        "memory_bytes": tensor.numel() * tensor.element_size(),
        "memory_mb": (tensor.numel() * tensor.element_size()) / 1024**2,
        "is_pinned": tensor.is_pinned() if hasattr(tensor, "is_pinned") else False,
    }

    # Add CUDA-specific info
    if tensor.is_cuda:
        info.update(
            {
                "cuda_device": tensor.device.index,
                "cuda_allocated": torch.cuda.memory_allocated(tensor.device),
                "cuda_reserved": torch.cuda.memory_reserved(tensor.device),
            }
        )

    return info


# =========================================================================
# Astronomical PyVista Zero-Copy Bridge
# =========================================================================


class AstronomicalPyVistaZeroCopyBridge:
    """PyVista bridge with astronomical context."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyVista tensor bridge.

        Args:
            device: Device for tensor operations ('cuda', 'cpu', etc.)
        """
        self.device = device or "cpu"
        self.plotter = None

    def ensure_cpu_contiguous(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on CPU and contiguous for PyVista astronomical operations."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def validate_astronomical_coordinates(self, tensor: Any) -> torch.Tensor:
        """
        Validate astronomical coordinate tensor for PyVista.

        Args:
            tensor: Input tensor with astronomical coordinates

        Returns:
            Validated tensor

        Raises:
            ValueError: If coordinates are invalid
        """
        if isinstance(tensor, torch.Tensor):
            coords = tensor
        elif hasattr(tensor, "cpu"):
            coords = tensor.cpu()
        else:
            coords = torch.tensor(tensor, dtype=torch.float32)

        # Validate coordinate dimensions
        if coords.dim() != 2:
            raise ValueError(
                f"Astronomical coordinates must be 2D, got {coords.dim()}D"
            )

        # Validate coordinate ranges for different systems
        if coords.shape[1] >= 2:
            ra, dec = coords[:, 0], coords[:, 1]

            # Check RA range (0-360 degrees)
            if torch.any(ra < 0) or torch.any(ra > 360):
                logger.warning("RA values outside 0-360 degree range detected")

            # Check Dec range (-90 to +90 degrees)
            if torch.any(dec < -90) or torch.any(dec > 90):
                logger.warning("Dec values outside -90 to +90 degree range detected")

        return coords

    def convert_astronomical_units(
        self, tensor: torch.Tensor, from_unit: str, to_unit: str
    ) -> torch.Tensor:
        """
        Convert astronomical units in tensor for PyVista.

        Args:
            tensor: Input tensor
            from_unit: Source unit (e.g., 'pc', 'kpc', 'Mpc')
            to_unit: Target unit (e.g., 'pc', 'kpc', 'Mpc')

        Returns:
            Tensor with converted units
        """
        # Define conversion factors
        unit_conversions = {
            "pc": 1.0,
            "kpc": 1000.0,
            "Mpc": 1000000.0,
            "ly": 3.26156,  # light years to parsecs
            "au": 4.84814e-6,  # astronomical units to parsecs
        }

        if from_unit == to_unit:
            return tensor

        if from_unit in unit_conversions and to_unit in unit_conversions:
            conversion_factor = unit_conversions[from_unit] / unit_conversions[to_unit]
            return tensor * conversion_factor
        else:
            logger.warning(f"Unknown unit conversion: {from_unit} to {to_unit}")
            return tensor

    def create_astronomical_sky_coordinates(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        distance: Optional[torch.Tensor] = None,
        frame: str = "icrs",
    ) -> SkyCoord:
        """
        Create Astropy SkyCoord from tensor data for PyVista.

        Args:
            ra: Right ascension tensor [degrees]
            dec: Declination tensor [degrees]
            distance: Distance tensor [parsecs]
            frame: Coordinate frame ('icrs', 'galactic')

        Returns:
            Astropy SkyCoord object
        """
        # Convert to numpy
        ra_np = ra.cpu().numpy()
        dec_np = dec.cpu().numpy()

        if distance is not None:
            distance_np = distance.cpu().numpy()
            return SkyCoord(
                ra=ra_np,
                dec=dec_np,
                distance=distance_np,
                unit=(u.Unit("deg"), u.Unit("deg"), u.Unit("pc")),
                frame=frame,
            )
        else:
            return SkyCoord(
                ra=ra_np, dec=dec_np, unit=(u.Unit("deg"), u.Unit("deg")), frame=frame
            )

    def to_pyvista(
        self,
        tensor: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        unit: str = "pc",
        **kwargs,
    ):
        """
        Convert astronomical tensor to PyVista with proper units using Enhanced-API.
        """
        with astronomical_pyvista_zero_copy_context("PyVista astronomical conversion"):
            # Optimize tensor
            optimized_tensor = optimize_astronomical_tensor_layout(tensor)
            # Validate coordinates
            coords = self.validate_astronomical_coordinates(optimized_tensor)
            coords_np = coords.cpu().numpy()
            # Prepare data dict for Enhanced-API
            data = {"coordinates": coords_np}
            if scalars is not None:
                data["scalars"] = scalars.cpu().numpy()
            mesh = to_pyvista(data, target_unit=unit)
            _astronomical_pyvista_mesh_manager.register_mesh(mesh, unit)
            logger.info(
                f"Created astronomical PyVista mesh with {len(coords_np)} points, unit: {unit}"
            )
            return mesh

    def to_pyvista_safe(
        self, tensor: torch.Tensor, scalars: Optional[torch.Tensor] = None, **kwargs
    ):
        """Safe version of to_pyvista with error handling."""
        try:
            return self.to_pyvista(tensor, scalars, **kwargs)
        except Exception as e:
            logger.error(f"Safe PyVista conversion failed: {e}")
            return None

    def cleanup_pyvista_mesh(self, mesh):
        """Clean up PyVista mesh."""
        _astronomical_pyvista_mesh_manager.safe_polydata_del(mesh)

    def cosmic_web_to_pyvista(
        self,
        spatial_tensor: Any,
        cluster_labels: Optional[np.ndarray] = None,
        density_field: Optional[torch.Tensor] = None,
        unit: str = "Mpc",
        **kwargs,
    ):
        """
        Convert cosmic web data to PyVista with astronomical context using Enhanced-API.
        """
        with astronomical_pyvista_zero_copy_context("Cosmic web PyVista conversion"):
            # Extract coordinates
            if (
                hasattr(spatial_tensor, "__getitem__")
                and "coordinates" in spatial_tensor
            ):
                coords = spatial_tensor["coordinates"]
            elif hasattr(spatial_tensor, "data"):
                coords = spatial_tensor.data
            else:
                coords = spatial_tensor
            coords = self.validate_astronomical_coordinates(coords)
            coords_np = coords.cpu().numpy()
            # Prepare data dict for Enhanced-API
            data = {"coordinates": coords_np}
            if cluster_labels is not None:
                data["cluster_labels"] = cluster_labels
            if density_field is not None:
                data["density"] = density_field.cpu().numpy()
            mesh = to_pyvista(data, target_unit=unit)
            _astronomical_pyvista_mesh_manager.register_mesh(mesh, unit)
            logger.info(
                f"Created cosmic web PyVista mesh with {len(coords_np)} points, unit: {unit}"
            )
            return mesh


@contextmanager
def astronomical_pyvista_mesh_context():
    """Context manager for astronomical PyVista mesh operations."""
    try:
        yield _astronomical_pyvista_mesh_manager
    finally:
        # Cleanup handled by manager
        pass


# =========================================================================
# Astronomical PyVista Utility Functions
# =========================================================================


def transfer_astronomical_to_pyvista(
    tensor: torch.Tensor, unit: str = "pc", **kwargs
) -> Any:
    """
    Transfer astronomical tensor to PyVista.

    Args:
        tensor: Astronomical tensor
        unit: Astronomical unit
        **kwargs: Additional parameters

    Returns:
        PyVista PolyData object
    """
    bridge = AstronomicalPyVistaZeroCopyBridge()
    return bridge.to_pyvista(tensor, unit=unit, **kwargs)


def quick_convert_astronomical_tensor_to_pyvista(
    tensor: torch.Tensor, unit: str = "pc", **kwargs
):
    """
    Quick conversion from astronomical tensor to PyVista.

    Args:
        tensor: Astronomical tensor
        unit: Astronomical unit
        **kwargs: Additional parameters

    Returns:
        PyVista mesh or None
    """
    bridge = AstronomicalPyVistaZeroCopyBridge()
    return bridge.to_pyvista(tensor, unit=unit, **kwargs)


def visualize_astronomical_cosmic_web_pyvista(
    spatial_tensor: Any,
    cluster_labels: Optional[np.ndarray] = None,
    density_field: Optional[torch.Tensor] = None,
    unit: str = "Mpc",
    **kwargs,
) -> Any:
    """
    Visualize cosmic web with PyVista and astronomical context.

    Args:
        spatial_tensor: Spatial coordinates tensor
        cluster_labels: Cluster assignments
        density_field: Density field tensor
        unit: Astronomical unit
        **kwargs: Additional parameters

    Returns:
        PyVista PolyData object
    """
    bridge = AstronomicalPyVistaZeroCopyBridge()
    return bridge.cosmic_web_to_pyvista(
        spatial_tensor, cluster_labels, density_field, unit=unit, **kwargs
    )


__all__ = [
    # Classes
    "AstronomicalPyVistaMeshManager",
    "AstronomicalPyVistaSyncConfig",
    "AstronomicalPyVistaZeroCopyBridge",
    # Functions
    "optimize_astronomical_tensor_layout",
    "get_astronomical_tensor_memory_info",
    "transfer_astronomical_to_pyvista",
    "quick_convert_astronomical_tensor_to_pyvista",
    "visualize_astronomical_cosmic_web_pyvista",
    # Context managers
    "astronomical_pyvista_zero_copy_context",
    "astronomical_pyvista_mesh_context",
]
