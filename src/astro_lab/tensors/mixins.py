"""
Core mixins for TensorDict functionality - Non-domain specific.

This module provides generic mixins that can be used across different
domains without astronomical or specific library assumptions.
"""

from typing import List, Optional, Tuple

import torch


class NormalizationMixin:
    """Generic data normalization mixin."""

    def normalize(
        self,
        method: str = "standard",
        data_key: str = "data",
        dim: int = -1,
        epsilon: float = 1e-8,
    ) -> "NormalizationMixin":
        """
        Normalize data using standard statistical methods.

        Args:
            method: 'standard', 'minmax', 'log'
            data_key: Key of the data tensor to normalize
            dim: Dimension to normalize over
            epsilon: Small constant for numerical stability

        Returns:
            Self with normalized data
        """
        if data_key not in self:
            raise ValueError(f"Data key '{data_key}' not found in TensorDict")

        data = self[data_key]

        if method == "standard":
            mean = torch.mean(data, dim=dim, keepdim=True)
            std = torch.std(data, dim=dim, keepdim=True)
            normalized = (data - mean) / (std + epsilon)
        elif method == "minmax":
            min_vals = torch.min(data, dim=dim, keepdim=True)[0]
            max_vals = torch.max(data, dim=dim, keepdim=True)[0]
            normalized = (data - min_vals) / (max_vals - min_vals + epsilon)
        elif method == "log":
            normalized = torch.log10(torch.clamp(data, min=epsilon))
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self[data_key] = normalized
        return self


class CoordinateConversionMixin:
    """Generic coordinate conversion utilities."""

    def to_spherical_coords(
        self, coords_key: str = "coordinates"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spherical.

        Args:
            coords_key: Key of the coordinates tensor [N, 3]

        Returns:
            Tuple of (azimuth, elevation, radius) tensors
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]
        if coords.shape[-1] != 3:
            raise ValueError("Coordinates must be 3D Cartesian")

        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

        # Radius
        radius = torch.norm(coords, dim=-1)

        # Azimuth (longitude) in radians [0, 2π)
        azimuth = torch.atan2(y, x)
        azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)

        # Elevation (latitude) in radians [-π/2, π/2]
        elevation = torch.asin(torch.clamp(z / (radius + 1e-10), -1, 1))

        return azimuth, elevation, radius

    def to_cartesian_coords(
        self, azimuth: torch.Tensor, elevation: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert spherical coordinates to Cartesian.

        Args:
            azimuth: Azimuth angle in radians
            elevation: Elevation angle in radians
            radius: Radius

        Returns:
            [N, 3] Cartesian coordinates
        """
        x = radius * torch.cos(elevation) * torch.cos(azimuth)
        y = radius * torch.cos(elevation) * torch.sin(azimuth)
        z = radius * torch.sin(elevation)

        return torch.stack([x, y, z], dim=-1)

    def euclidean_distance(
        self,
        other_coords: torch.Tensor,
        coords_key: str = "coordinates",
    ) -> torch.Tensor:
        """
        Calculate Euclidean distance between coordinates.

        Args:
            other_coords: [M, 3] Other coordinates
            coords_key: Key of the coordinates tensor

        Returns:
            [N, M] Distance matrix
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]

        # Pairwise distance calculation
        coords_expanded = coords.unsqueeze(1)  # [N, 1, 3]
        other_expanded = other_coords.unsqueeze(0)  # [1, M, 3]

        distances = torch.norm(coords_expanded - other_expanded, dim=-1)
        return distances


class FeatureExtractionMixin:
    """Generic feature extraction utilities."""

    def extract_statistical_features(
        self,
        data_key: str = "data",
        include_moments: bool = True,
        include_quantiles: bool = False,
    ) -> torch.Tensor:
        """
        Extract statistical features from data.

        Args:
            data_key: Key of the data tensor
            include_moments: Include statistical moments
            include_quantiles: Include quantile features

        Returns:
            [N, F] Feature tensor
        """
        if data_key not in self:
            raise ValueError(f"Data key '{data_key}' not found")

        data = self[data_key]
        features = []

        if include_moments:
            features.extend(
                [
                    torch.mean(data, dim=-1),
                    torch.std(data, dim=-1),
                    self._skewness(data),
                    self._kurtosis(data),
                ]
            )

        if include_quantiles:
            features.extend(
                [
                    torch.quantile(data, 0.25, dim=-1),
                    torch.quantile(data, 0.5, dim=-1),
                    torch.quantile(data, 0.75, dim=-1),
                ]
            )

        return (
            torch.stack(features, dim=-1) if features else torch.zeros(data.shape[0], 0)
        )

    def _skewness(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate skewness along given dimension."""
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        return torch.mean(normalized**3, dim=dim)

    def _kurtosis(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate kurtosis along given dimension."""
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        return torch.mean(normalized**4, dim=dim) - 3.0


class ValidationMixin:
    """Generic data validation utilities."""

    def validate_tensor_shape(
        self, key: str, expected_shape: Optional[Tuple[int, ...]] = None
    ) -> bool:
        """Validate tensor shape."""
        if key not in self:
            return False

        tensor = self[key]
        if not isinstance(tensor, torch.Tensor):
            return False

        if expected_shape is not None:
            # Check specific dimensions (-1 means any size)
            actual_shape = tensor.shape
            if len(actual_shape) != len(expected_shape):
                return False

            for actual, expected in zip(actual_shape, expected_shape):
                if expected != -1 and actual != expected:
                    return False

        return True

    def validate_finite_values(self, key: str) -> bool:
        """Validate that tensor contains finite values."""
        if key not in self:
            return False

        tensor = self[key]
        if not isinstance(tensor, torch.Tensor):
            return False

        return torch.isfinite(tensor).all()

    def validate_required_keys(self, required_keys: List[str]) -> bool:
        """Validate that required keys are present."""
        return all(key in self for key in required_keys)

    def validate_value_range(
        self, key: str, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> bool:
        """Validate that values are within specified range."""
        if key not in self:
            return False

        tensor = self[key]
        if not isinstance(tensor, torch.Tensor):
            return False

        if min_val is not None and torch.any(tensor < min_val):
            return False

        if max_val is not None and torch.any(tensor > max_val):
            return False

        return True
