"""Base sampler for graph construction and data loading.

Provides unified API for different sampling strategies including
k-NN, radius-based, cluster-based, and attention-based sampling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from sklearn.cluster import DBSCAN, KMeans
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import knn_graph


class AstroLabSampler(ABC):
    """Base class for all graph samplers.

    Defines unified API for graph construction and sampling:
    1. create_graph() - Build graph from data
    2. create_dataloader() - Create PyG DataLoader
    3. get_sampling_info() - Return sampling statistics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sampler with configuration.

        Args:
            config: Sampling configuration dict
        """
        self.config = config or {}
        self.sampling_stats: Dict[str, Any] = {}

    @abstractmethod
    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Union[Data, HeteroData]:
        """Create graph from coordinates and features.

        Args:
            coordinates: Node coordinates tensor [N, D]
            features: Node features tensor [N, F]
            **kwargs: Additional data (labels, masks, etc.)

        Returns:
            PyG Data or HeteroData object
        """

    @abstractmethod
    def create_dataloader(
        self,
        data: Union[Data, HeteroData, List[Data]],
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ) -> Union[DataLoader, NeighborLoader]:
        """Create DataLoader for training/inference.

        Args:
            data: PyG data object(s)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments

        Returns:
            PyG DataLoader
        """

    def get_sampling_info(self) -> Dict[str, Any]:
        """Get sampling statistics and information.

        Returns:
            Dictionary with sampling information
        """
        return {
            "config": self.config,
            "sampling_stats": self.sampling_stats,
            "sampler_type": self.__class__.__name__,
        }

    def validate_inputs(
        self, coordinates: torch.Tensor, features: torch.Tensor
    ) -> None:
        """Validate input tensors.

        Args:
            coordinates: Coordinate tensor
            features: Feature tensor

        Raises:
            ValueError: If inputs are invalid
        """
        if coordinates.dim() != 2:
            raise ValueError(f"Coordinates must be 2D, got {coordinates.dim()}D")

        if features.dim() != 2:
            raise ValueError(f"Features must be 2D, got {features.dim()}D")

        if coordinates.size(0) != features.size(0):
            raise ValueError(
                f"Coordinate and feature tensors must have same number of nodes: "
                f"{coordinates.size(0)} vs {features.size(0)}"
            )

        if torch.isnan(coordinates).any():
            raise ValueError("Coordinates contain NaN values")

        if torch.isnan(features).any():
            raise ValueError("Features contain NaN values")


class SpatialSamplerMixin:
    """Mixin providing spatial sampling utilities."""

    @staticmethod
    def calculate_edge_features(
        coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate edge features from coordinates.

        Args:
            coordinates: Node coordinates [N, D]
            edge_index: Edge indices [2, E]

        Returns:
            Edge features tensor [E, F]
        """
        source_coords = coordinates[edge_index[0]]
        target_coords = coordinates[edge_index[1]]
        distances = torch.norm(target_coords - source_coords, dim=1, keepdim=True)
        relative_pos = target_coords - source_coords
        edge_features = torch.cat([distances, relative_pos], dim=1)
        return edge_features

    @staticmethod
    def default_graph(
        coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create a default k-NN graph (k=2) for fallback usage."""

        edge_index = knn_graph(coordinates, k=2, loop=False)
        edge_attr = SpatialSamplerMixin.calculate_edge_features(coordinates, edge_index)
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        for key, value in kwargs.items():
            if key not in ["x", "pos", "edge_index", "edge_attr"]:
                setattr(data, key, value)
        return data


class ClusterSamplerMixin:
    """Mixin providing cluster-based sampling utilities."""

    @staticmethod
    def dbscan_clusters(
        coordinates: torch.Tensor, eps: float = 1.0, min_samples: int = 5
    ) -> torch.Tensor:
        """Perform DBSCAN clustering on coordinates.

        Args:
            coordinates: Node coordinates [N, D]
            eps: DBSCAN epsilon parameter
            min_samples: Minimum samples per cluster

        Returns:
            Cluster labels tensor [N]
        """

        coords_np = coordinates.cpu().numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(coords_np)
        return torch.tensor(labels, dtype=torch.long, device=coordinates.device)

    @staticmethod
    def kmeans_clusters(coordinates: torch.Tensor, n_clusters: int = 8) -> torch.Tensor:
        """Perform K-means clustering on coordinates.

        Args:
            coordinates: Node coordinates [N, D]
            n_clusters: Number of clusters

        Returns:
            Cluster labels tensor [N]
        """

        coords_np = coordinates.cpu().numpy()
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(coords_np)
        return torch.tensor(labels, dtype=torch.long, device=coordinates.device)


class AstronomicalSamplerMixin:
    """Mixin providing astronomy-specific sampling utilities."""

    @staticmethod
    def magnitude_based_sampling(
        features: torch.Tensor,
        mag_column_idx: int,
        bright_limit: float = 15.0,
        faint_limit: float = 20.0,
    ) -> Dict[str, torch.Tensor]:
        """Create magnitude-based node subsets.

        Args:
            features: Node features [N, F]
            mag_column_idx: Index of magnitude column
            bright_limit: Bright magnitude limit
            faint_limit: Faint magnitude limit

        Returns:
            Dictionary with masks for bright/intermediate/faint stars
        """
        magnitudes = features[:, mag_column_idx]

        return {
            "bright": magnitudes < bright_limit,
            "intermediate": (magnitudes >= bright_limit) & (magnitudes < faint_limit),
            "faint": magnitudes >= faint_limit,
        }

    @staticmethod
    def proper_motion_based_sampling(
        features: torch.Tensor,
        pmra_idx: int,
        pmdec_idx: int,
        high_pm_threshold: float = 50.0,
    ) -> Dict[str, torch.Tensor]:
        """Create proper motion-based node subsets.

        Args:
            features: Node features [N, F]
            pmra_idx: Index of RA proper motion column
            pmdec_idx: Index of Dec proper motion column
            high_pm_threshold: High proper motion threshold (mas/yr)

        Returns:
            Dictionary with masks for high/low proper motion stars
        """
        pmra = features[:, pmra_idx]
        pmdec = features[:, pmdec_idx]
        pm_total = torch.sqrt(pmra**2 + pmdec**2)

        return {
            "high_pm": pm_total > high_pm_threshold,
            "low_pm": pm_total <= high_pm_threshold,
        }

    @staticmethod
    def distance_based_sampling(
        coordinates: torch.Tensor, distance_ranges: List[tuple]
    ) -> Dict[str, torch.Tensor]:
        """Create distance-based node subsets.

        Args:
            coordinates: Node coordinates [N, 3] (assumed to include distance)
            distance_ranges: List of (min_dist, max_dist) tuples

        Returns:
            Dictionary with masks for different distance ranges
        """
        # Assume distance is the norm of coordinates or separate column
        if coordinates.size(1) == 3:
            distances = torch.norm(coordinates, dim=1)
        else:
            # Assume last column is distance
            distances = coordinates[:, -1]

        masks = {}
        for i, (min_dist, max_dist) in enumerate(distance_ranges):
            mask = (distances >= min_dist) & (distances < max_dist)
            masks[f"range_{i}"] = mask

        return masks
