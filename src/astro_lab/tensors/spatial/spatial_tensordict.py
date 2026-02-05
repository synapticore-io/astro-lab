"""
Spatial TensorDict for AstroLab
==============================

Spatial data handling leveraging native TensorDict features.
"""

from typing import Dict, List, Optional, Union

import torch
from tensordict import TensorDict

from ..base import AstroTensorDict


class SpatialTensorDict(AstroTensorDict):
    """
    Spatial tensor leveraging TensorDict features for astronomical coordinates.

    Key Features:
    - Native TensorDict batch operations
    - Efficient device management
    - Memory-mapped support for large datasets
    - torch.compile compatible operations
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        coordinate_system: str = "cartesian",
        unit: str = "pc",
        survey_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize spatial tensor with coordinates.

        Args:
            coordinates: Spatial coordinates [N, 3]
            coordinate_system: Coordinate system name
            unit: Distance unit
            survey_name: Name of astronomical survey
            **kwargs: Additional TensorDict arguments
        """
        # Ensure coordinates are tensor
        if not isinstance(coordinates, torch.Tensor):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)

        # Validate shape
        if coordinates.dim() != 2 or coordinates.size(1) != 3:
            raise ValueError(f"Coordinates must be [N, 3], got {coordinates.shape}")

        # Create data dict
        data = {
            "coordinates": coordinates,
            "coordinate_system": coordinate_system,
            "unit": unit,
        }

        # Add survey info if provided
        if survey_name:
            data["survey_name"] = survey_name

        # Initialize with batch_size from coordinates
        super().__init__(data, batch_size=torch.Size([coordinates.size(0)]), **kwargs)

    @property
    def coordinates(self) -> torch.Tensor:
        """Get coordinate tensor."""
        return self["coordinates"]

    @property
    def x(self) -> torch.Tensor:
        """X coordinates."""
        return self.coordinates[:, 0]

    @property
    def y(self) -> torch.Tensor:
        """Y coordinates."""
        return self.coordinates[:, 1]

    @property
    def z(self) -> torch.Tensor:
        """Z coordinates."""
        return self.coordinates[:, 2]

    @torch.compile(mode="reduce-overhead", dynamic=True)
    def compute_distances(self, other: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute distances with torch.compile optimization.

        Args:
            other: Other coordinates to compute distances to.
                  If None, compute distances from origin.

        Returns:
            Distance tensor
        """
        coords = self.coordinates

        if other is None:
            # Distance from origin
            return torch.norm(coords, dim=-1)
        else:
            # Pairwise distances
            return torch.cdist(coords, other)

    def build_knn_graph(self, k: int = 10) -> torch.Tensor:
        """Build k-nearest neighbor graph.

        Args:
            k: Number of neighbors

        Returns:
            Edge index tensor [2, num_edges]
        """
        from torch_geometric.nn import knn_graph

        edge_index = knn_graph(self.coordinates, k=k, loop=False, num_workers=4)

        # Store in TensorDict for reuse
        self["edge_index"] = edge_index
        self["k_neighbors"] = k

        return edge_index

    def build_radius_graph(self, r: float, max_num_neighbors: int = 64) -> torch.Tensor:
        """Build radius neighbor graph.

        Args:
            r: Radius in same units as coordinates
            max_num_neighbors: Maximum neighbors per node

        Returns:
            Edge index tensor [2, num_edges]
        """
        from torch_geometric.nn import radius_graph

        edge_index = radius_graph(
            self.coordinates,
            r=r,
            loop=False,
            max_num_neighbors=max_num_neighbors,
            num_workers=4,
        )

        # Store in TensorDict
        self["edge_index"] = edge_index
        self["radius"] = r

        return edge_index

    def to_pyg_data(self, node_features: Optional[torch.Tensor] = None) -> "Data":
        """Convert to PyTorch Geometric Data object.

        Args:
            node_features: Optional node feature matrix

        Returns:
            PyG Data object
        """
        from torch_geometric.data import Data

        # Use coordinates as default features
        if node_features is None:
            node_features = self.coordinates

        # Build edge index if not present
        if "edge_index" not in self:
            self.build_knn_graph(k=10)

        data = Data(
            x=node_features, edge_index=self["edge_index"], pos=self.coordinates
        )

        # Add metadata
        for key in ["coordinate_system", "unit", "survey_name"]:
            if key in self:
                setattr(data, key, self[key])

        return data

    def cosmic_web_clustering(
        self, eps: float = 10.0, min_samples: int = 5
    ) -> torch.Tensor:
        """Simple clustering for cosmic web analysis.

        Args:
            eps: Maximum distance between points
            min_samples: Minimum cluster size

        Returns:
            Cluster labels
        """
        from sklearn.cluster import DBSCAN

        # Use CPU for sklearn
        coords_cpu = self.coordinates.cpu().numpy()

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(coords_cpu)

        # Convert back to tensor
        labels_tensor = torch.tensor(labels, device=self.coordinates.device)

        # Store in TensorDict
        self["cluster_labels"] = labels_tensor
        self["clustering_eps"] = eps
        self["clustering_min_samples"] = min_samples

        return labels_tensor

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Union[TensorDict, Dict[str, torch.Tensor]]:
        """Extract spatial features.

        Args:
            feature_types: Types to extract ('spatial', 'graph')

        Returns:
            Feature TensorDict or dict
        """
        feature_td = TensorDict({}, batch_size=self.batch_size)

        if feature_types is None or "spatial" in feature_types:
            # Basic spatial features
            feature_td["coordinates"] = self.coordinates
            feature_td["distance_from_origin"] = self.compute_distances()
            feature_td["galactic_height"] = torch.abs(self.z)

        if feature_types is None or "graph" in feature_types:
            # Graph features if available
            if "edge_index" in self:
                feature_td["edge_index"] = self["edge_index"]

                # Compute edge features
                row, col = self["edge_index"]
                edge_vec = self.coordinates[row] - self.coordinates[col]
                feature_td["edge_length"] = torch.norm(edge_vec, dim=-1)
                feature_td["edge_direction"] = edge_vec / (
                    feature_td["edge_length"].unsqueeze(-1) + 1e-8
                )

        return feature_td if not kwargs.get("as_dict", False) else feature_td.to_dict()

    @classmethod
    def from_survey_data(
        cls,
        survey_df: "DataFrame",
        coord_cols: List[str] = ["x", "y", "z"],
        survey_name: Optional[str] = None,
        **kwargs,
    ) -> "SpatialTensorDict":
        """Create from survey dataframe.

        Args:
            survey_df: Survey data with coordinates
            coord_cols: Column names for coordinates
            survey_name: Survey identifier

        Returns:
            SpatialTensorDict instance
        """
        # Extract coordinates
        coords_data = survey_df[coord_cols].to_numpy()
        coords = torch.tensor(coords_data, dtype=torch.float32)

        # Create instance
        spatial_td = cls(coordinates=coords, survey_name=survey_name, **kwargs)

        # Add other numeric columns as features
        numeric_cols = [c for c in survey_df.columns if c not in coord_cols]
        for col in numeric_cols:
            try:
                data = torch.tensor(survey_df[col].to_numpy(), dtype=torch.float32)
                if data.shape[0] == coords.shape[0]:
                    spatial_td[col] = data
            except:
                continue

        return spatial_td

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpatialTensorDict(n_objects={self.n_objects}, "
            f"coordinate_system='{self.get('coordinate_system', 'unknown')}', "
            f"device={self.device})"
        )
