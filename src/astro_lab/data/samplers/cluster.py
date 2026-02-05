"""Cluster-based samplers for astronomical data.

Implements clustering algorithms for graph construction.
"""

from typing import List, Optional, Union

import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric.nn import knn_graph

from .base import AstroLabSampler, ClusterSamplerMixin, SpatialSamplerMixin


class ClusterSampler(AstroLabSampler, ClusterSamplerMixin, SpatialSamplerMixin):
    """Cluster-based graph sampler using METIS or spectral clustering."""

    def __init__(
        self,
        num_parts: int = 100,
        recursive: bool = True,
        batch_size: int = 10,
        num_workers: int = 0,
        **kwargs,
    ):
        """Initialize cluster sampler.

        Args:
            num_parts: Number of partitions/clusters
            recursive: Use recursive clustering
            batch_size: Number of clusters per batch
            num_workers: Number of data loading workers
            **kwargs: Additional config
        """
        super().__init__(
            {
                "num_parts": num_parts,
                "recursive": recursive,
                "batch_size": batch_size,
                "num_workers": num_workers,
                **kwargs,
            }
        )
        self.num_parts = num_parts
        self.recursive = recursive
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cluster_data = None

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create graph with clustering preparation.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data

        Returns:
            PyG Data object
        """
        self.validate_inputs(coordinates, features)

        # Create initial k-NN graph for connectivity
        k = min(20, coordinates.size(0) // 10)

        edge_index = knn_graph(coordinates, k=k, loop=False)

        # Calculate edge weights based on distance
        edge_attr = self.calculate_edge_features(coordinates, edge_index)
        edge_weight = 1.0 / (edge_attr[:, 0] + 1e-6)  # Inverse distance

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            setattr(data, key, value)

        # Create ClusterData for efficient partitioning
        self.cluster_data = ClusterData(
            data, num_parts=self.num_parts, recursive=self.recursive, log=True
        )

        # Update stats
        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "num_clusters": self.num_parts,
                "avg_cluster_size": coordinates.size(0) / self.num_parts,
            }
        )

        return data

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> ClusterLoader:
        """Create ClusterLoader for efficient cluster-based sampling.

        Args:
            data: PyG Data object
            batch_size: Override default batch size
            shuffle: Whether to shuffle clusters
            **kwargs: Additional DataLoader args

        Returns:
            ClusterLoader instance
        """
        if self.cluster_data is None:
            # Create ClusterData if not already done
            self.cluster_data = ClusterData(
                data, num_parts=self.num_parts, recursive=self.recursive, log=False
            )

        batch_size = batch_size or self.batch_size

        return ClusterLoader(
            self.cluster_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            **kwargs,
        )


class DBSCANClusterSampler(AstroLabSampler, ClusterSamplerMixin, SpatialSamplerMixin):
    """DBSCAN-based clustering for astronomical structures."""

    def __init__(
        self, eps: float = 10.0, min_samples: int = 5, batch_size: int = 32, **kwargs
    ):
        """Initialize DBSCAN cluster sampler.

        Args:
            eps: DBSCAN epsilon (maximum distance)
            min_samples: Minimum samples per cluster
            batch_size: Batch size for DataLoader
            **kwargs: Additional config
        """
        super().__init__(
            {"eps": eps, "min_samples": min_samples, "batch_size": batch_size, **kwargs}
        )
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create graph based on DBSCAN clustering.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data

        Returns:
            PyG Data object with cluster information
        """
        self.validate_inputs(coordinates, features)

        # Perform DBSCAN clustering
        cluster_labels = self.dbscan_clusters(
            coordinates, eps=self.eps, min_samples=self.min_samples
        )

        # Create edges within clusters
        edge_list = []
        unique_clusters = torch.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue

            # Get nodes in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_nodes = torch.where(cluster_mask)[0]

            # Create fully connected subgraph for small clusters
            if len(cluster_nodes) <= 20:
                # Vectorized creation of fully connected graph
                n_cluster = len(cluster_nodes)
                # Create all pairs using meshgrid
                src_idx = torch.arange(n_cluster, device=cluster_nodes.device)
                src_idx = src_idx.unsqueeze(1).expand(n_cluster, n_cluster)
                dst_idx = torch.arange(n_cluster, device=cluster_nodes.device)
                dst_idx = dst_idx.unsqueeze(0).expand(n_cluster, n_cluster)

                # Remove self-loops and get upper triangle indices
                mask = src_idx < dst_idx
                src_local = src_idx[mask]
                dst_local = dst_idx[mask]

                # Map to global node indices
                src_global = cluster_nodes[src_local]
                dst_global = cluster_nodes[dst_local]

                # Add bidirectional edges
                edges = torch.stack(
                    [
                        torch.cat([src_global, dst_global]),
                        torch.cat([dst_global, src_global]),
                    ],
                    dim=0,
                )
                edge_list.append(edges)
            else:
                # For larger clusters, use k-NN within cluster
                cluster_coords = coordinates[cluster_nodes]
                local_edges = knn_graph(cluster_coords, k=10)

                # Map back to global indices
                global_edges = cluster_nodes[local_edges]
                edge_list.append(global_edges)

        if edge_list:
            edge_index = torch.cat(edge_list, dim=1)
        else:
            # Fallback to k-NN if no clusters found
            edge_index = knn_graph(coordinates, k=10)

        # Calculate edge features
        edge_attr = self.calculate_edge_features(coordinates, edge_index)

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cluster=cluster_labels,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            setattr(data, key, value)

        # Update stats
        n_clusters = len(unique_clusters[unique_clusters != -1])
        n_noise = (cluster_labels == -1).sum().item()

        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "num_clusters": n_clusters,
                "num_noise_points": n_noise,
                "clustering_eps": self.eps,
            }
        )

        return data

    def create_dataloader(
        self,
        data: Union[Data, List[Data]],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader with cluster-aware batching."""
        batch_size = batch_size or self.batch_size

        # If we have cluster information, we could do smarter batching
        # For now, standard DataLoader
        return DataLoader(
            data if isinstance(data, list) else [data],
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )


class HierarchicalClusterSampler(AstroLabSampler, ClusterSamplerMixin):
    """Hierarchical clustering for multi-scale astronomical structures."""

    def __init__(
        self,
        distance_thresholds: List[float] = [10.0, 50.0, 200.0],
        linkage: str = "single",
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize hierarchical cluster sampler.

        Args:
            distance_thresholds: Distance thresholds for different levels
            linkage: Linkage criterion ('single', 'complete', 'average')
            batch_size: Batch size for DataLoader
            **kwargs: Additional config
        """
        super().__init__(
            {
                "distance_thresholds": distance_thresholds,
                "linkage": linkage,
                "batch_size": batch_size,
                **kwargs,
            }
        )
        self.distance_thresholds = sorted(distance_thresholds)
        self.linkage = linkage
        self.batch_size = batch_size

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create hierarchical graph structure.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data

        Returns:
            PyG Data object with hierarchical clustering
        """
        self.validate_inputs(coordinates, features)

        # Compute hierarchical clustering
        coords_np = coordinates.cpu().numpy()
        condensed_dist = pdist(coords_np)
        Z = linkage(condensed_dist, method=self.linkage)

        # Get clusters at different thresholds
        cluster_levels = {}
        edge_lists = []

        for i, threshold in enumerate(self.distance_thresholds):
            clusters = fcluster(Z, threshold, criterion="distance")
            cluster_tensor = torch.tensor(clusters - 1, dtype=torch.long)
            cluster_levels[f"cluster_level_{i}"] = cluster_tensor

            # Create edges within clusters at this level
            unique_clusters = torch.unique(cluster_tensor)

            for cluster_id in unique_clusters:
                cluster_mask = cluster_tensor == cluster_id
                cluster_nodes = torch.where(cluster_mask)[0]

                if len(cluster_nodes) > 1 and len(cluster_nodes) <= 30:
                    # Small clusters: fully connected
                    for i in range(len(cluster_nodes)):
                        for j in range(i + 1, len(cluster_nodes)):
                            edge_lists.append(
                                torch.tensor(
                                    [
                                        [cluster_nodes[i], cluster_nodes[j]],
                                        [cluster_nodes[j], cluster_nodes[i]],
                                    ],
                                    dtype=torch.long,
                                )
                            )

        # Combine all edges
        if edge_lists:
            edge_index = torch.cat(edge_lists, dim=1)
        else:
            # Fallback to k-NN
            edge_index = knn_graph(coordinates, k=10)

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
        )

        # Add cluster levels
        for level_name, cluster_tensor in cluster_levels.items():
            setattr(data, level_name, cluster_tensor)

        # Add additional attributes
        for key, value in kwargs.items():
            setattr(data, key, value)

        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "num_levels": len(self.distance_thresholds),
                "linkage": self.linkage,
            }
        )

        return data

    def create_dataloader(
        self,
        data: Union[Data, List[Data]],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader."""
        batch_size = batch_size or self.batch_size

        return DataLoader(
            data if isinstance(data, list) else [data],
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
