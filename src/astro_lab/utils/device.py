"""Device detection and management utilities for AstroLab.

This module provides centralized device detection to avoid repeated CUDA checks
throughout the codebase, improving performance and maintainability.
"""

from typing import Union

import torch

# Cache the device detection results
_cached_cuda_available: bool = None
_cached_default_device: str = None


def is_cuda_available() -> bool:
    """Check if CUDA is available with caching.

    Returns:
        True if CUDA is available, False otherwise
    """
    global _cached_cuda_available
    if _cached_cuda_available is None:
        _cached_cuda_available = torch.cuda.is_available()
    return _cached_cuda_available


def get_default_device() -> str:
    """Get the default device (GPU if available, else CPU).

    Returns:
        Device string: 'cuda' or 'cpu'
    """
    global _cached_default_device
    if _cached_default_device is None:
        _cached_default_device = "cuda" if is_cuda_available() else "cpu"
    return _cached_default_device


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """Get a torch device, with smart defaults.

    Args:
        device: Optional device specification. If None, uses default device.

    Returns:
        torch.device instance
    """
    if device is None:
        return torch.device(get_default_device())
    return torch.device(device)


def reset_device_cache():
    """Reset the device detection cache.

    This is useful for testing or when device availability changes.
    """
    global _cached_cuda_available, _cached_default_device
    _cached_cuda_available = None
    _cached_default_device = None
