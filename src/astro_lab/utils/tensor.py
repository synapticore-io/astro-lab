"""Tensor utility functions for AstroLab.

This module provides helper functions for working with tensors and TensorDicts.
"""

from typing import Union

import torch


def extract_coordinates(
    coordinates: Union[torch.Tensor, "SpatialTensorDict"],
) -> torch.Tensor:
    """Extract coordinate tensor from various input types.

    This is a common pattern used throughout the codebase to handle
    both raw tensors and SpatialTensorDict inputs.

    Args:
        coordinates: Either a torch.Tensor or SpatialTensorDict containing coordinates

    Returns:
        Coordinate tensor of shape [N, 3]

    Examples:
        >>> coords = torch.randn(100, 3)
        >>> result = extract_coordinates(coords)
        >>> assert result.shape == (100, 3)

        >>> from astro_lab.tensors import SpatialTensorDict
        >>> spatial = SpatialTensorDict(coords)
        >>> result = extract_coordinates(spatial)
        >>> assert result.shape == (100, 3)
    """
    # Handle SpatialTensorDict
    if hasattr(coordinates, "coordinates"):
        return coordinates.coordinates

    # Handle dict-like TensorDict with "coordinates" key
    if hasattr(coordinates, "__getitem__"):
        try:
            return coordinates["coordinates"]
        except (KeyError, TypeError):
            pass

    # Assume it's already a tensor
    return coordinates
