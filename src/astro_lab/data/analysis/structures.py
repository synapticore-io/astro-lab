"""
Structure Analysis for Astronomical Data
=======================================

Advanced structure detection and analysis with TensorDict integration.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected

from astro_lab.tensors import SpatialTensorDict
from astro_lab.utils.device import get_default_device
from astro_lab.utils.tensor import extract_coordinates

logger = logging.getLogger(__name__)


class FilamentDetector:
    """
    Advanced filament detection using geometric and topological analysis.

    Features:
    - Multi-scale filament detection
    - Topological persistence analysis
    - TensorDict integration
    - GPU acceleration
    """

    def __init__(
        self,
        device: str = None,
        min_filament_length: float = 2.0,
        anisotropy_threshold: float = 0.7,
    ):
        """Initialize filament detector."""
        if device is None:
            device = get_default_device()
        self.device = device
        self.min_filament_length = min_filament_length
        self.anisotropy_threshold = anisotropy_threshold

        logger.info(f"🧵 FilamentDetector initialized on {self.device}")

    def detect_filaments(
        self,
        coordinates: Union[torch.Tensor, SpatialTensorDict],
        density_field: Optional[torch.Tensor] = None,
        scales: Optional[List[float]] = None,
    ) -> Dict:
        """
        Detect filamentary structures in the data.

        Args:
            coordinates: Object coordinates [N, 3]
            density_field: Optional density field [N]
            scales: Analysis scales

        Returns:
            Filament detection results
        """
        # Handle TensorDict input using utility function
        coords = extract_coordinates(coordinates)
        coords = coords.to(self.device)

        if density_field is not None:
            density_field = density_field.to(self.device)

        if scales is None:
            scales = [1.0, 2.0, 5.0]

        logger.info(
            f"🧵 Filament detection: {coords.size(0)} points, {len(scales)} scales"
        )

        results = {}

        for scale in scales:
            scale_results = self._detect_at_scale(coords, density_field, scale)
            results[f"scale_{scale:.1f}"] = scale_results

        # Combine multi-scale results
        combined = self._combine_filament_results(results)

        return {
            "multi_scale": results,
            "combined": combined,
            "coordinates": coords,
            "scales": scales,
        }

    def _detect_at_scale(
        self,
        coordinates: torch.Tensor,
        density_field: Optional[torch.Tensor],
        scale: float,
    ) -> Dict:
        """Detect filaments at a specific scale."""
        n_points = coordinates.size(0)

        # Build connectivity graph
        edge_index = radius_graph(
            coordinates,
            r=scale,
            batch=None,
            loop=False,
            max_num_neighbors=min(100, n_points),
        )

        # Make undirected
        edge_index = to_undirected(edge_index, num_nodes=n_points)

        # Calculate local density if not provided
        if density_field is None:
            density_field = self._calculate_local_density(coordinates, edge_index)

        # Calculate geometric features
        anisotropy = self._calculate_anisotropy(coordinates, edge_index)
        curvature = self._calculate_curvature(coordinates, edge_index)

        # Identify filament candidates
        filament_mask = self._identify_filament_candidates(
            density_field, anisotropy, curvature, scale
        )

        # Extract filament structures
        filaments = self._extract_filament_structures(
            coordinates, edge_index, filament_mask, scale
        )

        return {
            "scale": scale,
            "filaments": filaments,
            "anisotropy": anisotropy,
            "curvature": curvature,
            "density_field": density_field,
            "filament_mask": filament_mask,
        }

    def _calculate_local_density(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local density using connectivity."""
        n_points = coordinates.size(0)
        device = coordinates.device

        # Count neighbors for each point
        neighbor_counts = torch.zeros(n_points, device=device)
        src = edge_index[0]
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        # Normalize by maximum possible neighbors
        max_neighbors = neighbor_counts.max()
        if max_neighbors > 0:
            density = neighbor_counts / max_neighbors
        else:
            density = torch.zeros_like(neighbor_counts)

        return density

    def _calculate_anisotropy(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local anisotropy using neighbor distribution (vectorized)."""
        n_points = coordinates.size(0)
        device = coordinates.device

        anisotropy = torch.zeros(n_points, device=device)

        # Vectorized approach: process all edges at once
        src, dst = edge_index[0], edge_index[1]

        # Calculate direction vectors for all edges
        directions = coordinates[dst] - coordinates[src]
        directions = F.normalize(directions, dim=1, eps=1e-8)

        # Group by source node using scatter operations
        # Count neighbors per node
        neighbor_counts = torch.zeros(n_points, device=device, dtype=torch.long)
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src))

        # Calculate mean direction per node
        mean_directions = torch.zeros(n_points, 3, device=device)
        mean_directions.scatter_add_(
            0, src.unsqueeze(1).expand_as(directions), directions
        )
        valid_mask = neighbor_counts > 0
        mean_directions[valid_mask] /= neighbor_counts[valid_mask].unsqueeze(1).float()

        # Calculate variance for each edge
        mean_expanded = mean_directions[src]
        variances = ((directions - mean_expanded) ** 2).sum(dim=1)

        # Aggregate variance per node
        variance_sum = torch.zeros(n_points, device=device)
        variance_sum.scatter_add_(0, src, variances)

        # Average variance (anisotropy) for nodes with multiple neighbors
        multi_neighbor_mask = neighbor_counts > 1
        anisotropy[multi_neighbor_mask] = (
            variance_sum[multi_neighbor_mask]
            / neighbor_counts[multi_neighbor_mask].float()
        )

        return anisotropy

    def _calculate_curvature(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local curvature using neighbor positions (vectorized)."""
        n_points = coordinates.size(0)
        device = coordinates.device

        curvature = torch.zeros(n_points, device=device)

        # Vectorized approach
        src, dst = edge_index[0], edge_index[1]

        # Calculate distances for all edges
        distances = torch.norm(coordinates[dst] - coordinates[src], dim=1)

        # Count neighbors per node
        neighbor_counts = torch.zeros(n_points, device=device, dtype=torch.long)
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src))

        # Calculate mean distance per node
        distance_sum = torch.zeros(n_points, device=device)
        distance_sum.scatter_add_(0, src, distances)
        mean_distance = torch.zeros(n_points, device=device)
        valid_mask = neighbor_counts > 0
        mean_distance[valid_mask] = (
            distance_sum[valid_mask] / neighbor_counts[valid_mask].float()
        )

        # Calculate squared deviations
        mean_expanded = mean_distance[src]
        sq_deviations = (distances - mean_expanded) ** 2

        # Sum squared deviations per node
        sq_dev_sum = torch.zeros(n_points, device=device)
        sq_dev_sum.scatter_add_(0, src, sq_deviations)

        # Calculate standard deviation and curvature for nodes with 2+ neighbors
        multi_neighbor_mask = neighbor_counts >= 2
        std_distance = torch.sqrt(
            sq_dev_sum[multi_neighbor_mask]
            / neighbor_counts[multi_neighbor_mask].float()
        )
        curvature[multi_neighbor_mask] = std_distance / (
            mean_distance[multi_neighbor_mask] + 1e-8
        )

        return curvature

    def _identify_filament_candidates(
        self,
        density_field: torch.Tensor,
        anisotropy: torch.Tensor,
        curvature: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Identify filament candidate points."""
        # Filament criteria: high anisotropy, medium density, low curvature
        filament_mask = (
            (anisotropy > self.anisotropy_threshold)
            & (density_field > 0.3)
            & (density_field < 0.8)
            & (curvature < 0.5)
        )

        return filament_mask

    def _extract_filament_structures(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        filament_mask: torch.Tensor,
        scale: float,
    ) -> List[Dict]:
        """Extract connected filament structures."""
        n_points = coordinates.size(0)
        device = coordinates.device

        # Find connected filament components
        filament_edges = edge_index[
            :, filament_mask[edge_index[0]] & filament_mask[edge_index[1]]
        ]

        if filament_edges.size(1) > 0:
            filament_labels = self._connected_components(filament_edges, n_points)
        else:
            filament_labels = torch.full(
                (n_points,), -1, dtype=torch.long, device=device
            )

        # Extract individual filaments
        filaments = []
        for label in torch.unique(filament_labels):
            if label < 0:
                continue

            filament_mask_label = filament_labels == label
            filament_coords = coordinates[filament_mask_label]

            if len(filament_coords) >= 3:
                # Calculate filament properties
                length = self._calculate_filament_length(filament_coords)
                thickness = self._calculate_filament_thickness(filament_coords)

                if length > self.min_filament_length:
                    filaments.append(
                        {
                            "label": label.item(),
                            "coordinates": filament_coords,
                            "length": length,
                            "thickness": thickness,
                            "n_points": len(filament_coords),
                            "point_indices": filament_mask_label.nonzero().squeeze(),
                        }
                    )

        return filaments

    def _connected_components(
        self, edge_index: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        """Find connected components using Union-Find."""
        device = edge_index.device

        # Initialize each node as its own component
        parent = torch.arange(n_nodes, device=device)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union connected nodes
        for i in range(edge_index.size(1)):
            union(edge_index[0, i], edge_index[1, i])

        # Relabel components
        labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
        component_map = {}
        next_label = 0

        for i in range(n_nodes):
            root = find(i)
            if root not in component_map:
                component_map[root] = next_label
                next_label += 1
            labels[i] = component_map[root]

        return labels

    def _calculate_filament_length(self, coordinates: torch.Tensor) -> float:
        """Calculate filament length."""
        if len(coordinates) < 2:
            return 0.0

        # Use maximum pairwise distance as length estimate
        distances = torch.cdist(coordinates, coordinates)
        max_distance = distances.max().item()

        return max_distance

    def _calculate_filament_thickness(self, coordinates: torch.Tensor) -> float:
        """Calculate filament thickness."""
        if len(coordinates) < 3:
            return 0.0

        # Use minimum spanning tree edge length as thickness estimate
        distances = torch.cdist(coordinates, coordinates)
        # Set diagonal to infinity
        distances.fill_diagonal_(float("inf"))
        min_distances = distances.min(dim=1)[0]

        return min_distances.mean().item()

    def _combine_filament_results(self, results: Dict) -> Dict:
        """Combine filament results from multiple scales."""
        combined = {
            "filaments": [],
            "statistics": {},
        }

        # Aggregate filaments across scales
        for scale_name, scale_result in results.items():
            filaments = scale_result["filaments"]
            for filament in filaments:
                filament["scale"] = scale_result["scale"]
                combined["filaments"].append(filament)

        # Calculate statistics
        if combined["filaments"]:
            lengths = [f["length"] for f in combined["filaments"]]
            thicknesses = [f["thickness"] for f in combined["filaments"]]
            n_points = [f["n_points"] for f in combined["filaments"]]

            combined["statistics"] = {
                "n_filaments": len(combined["filaments"]),
                "mean_length": sum(lengths) / len(lengths),
                "mean_thickness": sum(thicknesses) / len(thicknesses),
                "mean_points": sum(n_points) / len(n_points),
                "total_length": sum(lengths),
            }
        else:
            combined["statistics"] = {
                "n_filaments": 0,
                "mean_length": 0.0,
                "mean_thickness": 0.0,
                "mean_points": 0.0,
                "total_length": 0.0,
            }

        return combined


class StructureAnalyzer:
    """
    Comprehensive structure analysis for astronomical data.

    Features:
    - Multi-scale structure analysis
    - Hierarchical structure detection
    - TensorDict integration
    - Statistical analysis
    """

    def __init__(
        self,
        device: str = None,
        max_structures: int = 1000,
    ):
        """Initialize structure analyzer."""
        if device is None:
            device = get_default_device()
        self.device = device
        self.max_structures = max_structures

        logger.info(f"🏗️ StructureAnalyzer initialized on {self.device}")

    def analyze_structures(
        self,
        coordinates: Union[torch.Tensor, SpatialTensorDict],
        density_field: Optional[torch.Tensor] = None,
        scales: Optional[List[float]] = None,
    ) -> Dict:
        """
        Analyze structures at multiple scales.

        Args:
            coordinates: Object coordinates [N, 3]
            density_field: Optional density field [N]
            scales: Analysis scales

        Returns:
            Structure analysis results
        """
        # Handle TensorDict input using utility function
        coords = extract_coordinates(coordinates)
        coords = coords.to(self.device)

        if density_field is not None:
            density_field = density_field.to(self.device)

        if scales is None:
            scales = [1.0, 2.0, 5.0, 10.0]

        logger.info(
            f"🏗️ Structure analysis: {coords.size(0)} points, {len(scales)} scales"
        )

        results = {}

        for scale in scales:
            scale_results = self._analyze_at_scale(coords, density_field, scale)
            results[f"scale_{scale:.1f}"] = scale_results

        # Combine multi-scale results
        combined = self._combine_structure_results(results)

        return {
            "multi_scale": results,
            "combined": combined,
            "coordinates": coords,
            "scales": scales,
        }

    def _analyze_at_scale(
        self,
        coordinates: torch.Tensor,
        density_field: Optional[torch.Tensor],
        scale: float,
    ) -> Dict:
        """Analyze structures at a specific scale."""
        n_points = coordinates.size(0)

        # Build connectivity graph
        edge_index = radius_graph(
            coordinates,
            r=scale,
            batch=None,
            loop=False,
            max_num_neighbors=min(100, n_points),
        )

        # Make undirected
        edge_index = to_undirected(edge_index, num_nodes=n_points)

        # Calculate local density if not provided
        if density_field is None:
            density_field = self._calculate_local_density(coordinates, edge_index)

        # Identify different structure types
        clusters = self._identify_clusters(
            coordinates, edge_index, density_field, scale
        )
        voids = self._identify_voids(coordinates, density_field, scale)
        walls = self._identify_walls(coordinates, edge_index, density_field, scale)

        return {
            "scale": scale,
            "clusters": clusters,
            "voids": voids,
            "walls": walls,
            "density_field": density_field,
            "connectivity": edge_index,
        }

    def _calculate_local_density(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local density using connectivity."""
        n_points = coordinates.size(0)
        device = coordinates.device

        # Count neighbors for each point
        neighbor_counts = torch.zeros(n_points, device=device)
        src = edge_index[0]
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        # Normalize by maximum possible neighbors
        max_neighbors = neighbor_counts.max()
        if max_neighbors > 0:
            density = neighbor_counts / max_neighbors
        else:
            density = torch.zeros_like(neighbor_counts)

        return density

    def _identify_clusters(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        density_field: torch.Tensor,
        scale: float,
    ) -> List[Dict]:
        """Identify cluster structures."""
        # High density regions
        cluster_mask = density_field > 0.7

        # Find connected cluster components
        cluster_edges = edge_index[
            :, cluster_mask[edge_index[0]] & cluster_mask[edge_index[1]]
        ]

        if cluster_edges.size(1) > 0:
            cluster_labels = self._connected_components(
                cluster_edges, coordinates.size(0)
            )
        else:
            cluster_labels = torch.full(
                (coordinates.size(0),), -1, dtype=torch.long, device=coordinates.device
            )

        # Extract clusters
        clusters = []
        for label in torch.unique(cluster_labels):
            if label < 0:
                continue

            cluster_mask_label = cluster_labels == label
            cluster_coords = coordinates[cluster_mask_label]

            if len(cluster_coords) >= 5:
                # Calculate cluster properties
                center = cluster_coords.mean(dim=0)
                radius = torch.norm(cluster_coords - center, dim=1).max().item()
                mass = len(cluster_coords)

                clusters.append(
                    {
                        "label": label.item(),
                        "coordinates": cluster_coords,
                        "center": center,
                        "radius": radius,
                        "mass": mass,
                        "point_indices": cluster_mask_label.nonzero().squeeze(),
                    }
                )

        return clusters

    def _identify_voids(
        self, coordinates: torch.Tensor, density_field: torch.Tensor, scale: float
    ) -> List[Dict]:
        """Identify void structures."""
        # Low density regions
        void_mask = density_field < 0.3

        # Find connected void components
        void_coords = coordinates[void_mask]

        if void_coords.size(0) > 0:
            # Use distance-based clustering for voids
            void_labels = self._cluster_voids(void_coords, scale)
        else:
            void_labels = torch.empty(0, dtype=torch.long, device=coordinates.device)

        # Extract voids
        voids = []
        for label in torch.unique(void_labels):
            if label >= 0:
                void_coords_label = void_coords[void_labels == label]

                if len(void_coords_label) >= 10:
                    # Calculate void properties
                    center = void_coords_label.mean(dim=0)
                    radius = torch.norm(void_coords_label - center, dim=1).max().item()
                    volume = len(void_coords_label)

                    voids.append(
                        {
                            "label": label.item(),
                            "coordinates": void_coords_label,
                            "center": center,
                            "radius": radius,
                            "volume": volume,
                        }
                    )

        return voids

    def _identify_walls(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        density_field: torch.Tensor,
        scale: float,
    ) -> List[Dict]:
        """Identify wall structures."""
        # Medium density, planar regions
        wall_mask = (density_field > 0.4) & (density_field < 0.7)

        # Calculate planarity
        planarity = self._calculate_planarity(coordinates, edge_index)
        wall_mask = wall_mask & (planarity > 0.6)

        # Find connected wall components
        wall_edges = edge_index[:, wall_mask[edge_index[0]] & wall_mask[edge_index[1]]]

        if wall_edges.size(1) > 0:
            wall_labels = self._connected_components(wall_edges, coordinates.size(0))
        else:
            wall_labels = torch.full(
                (coordinates.size(0),), -1, dtype=torch.long, device=coordinates.device
            )

        # Extract walls
        walls = []
        for label in torch.unique(wall_labels):
            if label < 0:
                continue

            wall_mask_label = wall_labels == label
            wall_coords = coordinates[wall_mask_label]

            if len(wall_coords) >= 8:
                # Calculate wall properties
                center = wall_coords.mean(dim=0)
                area = len(wall_coords)
                thickness = self._calculate_wall_thickness(wall_coords)

                walls.append(
                    {
                        "label": label.item(),
                        "coordinates": wall_coords,
                        "center": center,
                        "area": area,
                        "thickness": thickness,
                        "point_indices": wall_mask_label.nonzero().squeeze(),
                    }
                )

        return walls

    def _calculate_planarity(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local planarity using neighbor distribution (vectorized).

        Note: This function still uses a loop for eigenvalue calculations
        but is optimized where possible. Full vectorization of eigenvalue
        computation across all nodes would be complex and may not provide
        significant speedup for typical graph sizes.
        """
        n_points = coordinates.size(0)
        device = coordinates.device

        planarity = torch.zeros(n_points, device=device)

        # Pre-compute neighbor counts to identify nodes with sufficient neighbors
        src = edge_index[0]
        neighbor_counts = torch.zeros(n_points, device=device, dtype=torch.long)
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src))

        # Only process nodes with 3+ neighbors
        valid_nodes = torch.where(neighbor_counts > 2)[0]

        for i in valid_nodes:
            # Find neighbors
            neighbor_mask = edge_index[0] == i
            neighbors = edge_index[1, neighbor_mask]

            # Calculate local covariance matrix
            neighbor_positions = coordinates[neighbors]
            center = coordinates[i]
            relative_positions = neighbor_positions - center

            # Compute covariance matrix
            cov_matrix = torch.matmul(relative_positions.T, relative_positions) / len(
                relative_positions
            )

            # Calculate eigenvalues
            eigenvals = torch.linalg.eigvals(cov_matrix).real
            eigenvals = torch.sort(eigenvals, descending=True)[0]

            # Planarity = (λ1 - λ2) / (λ1 + λ2 + λ3)
            if eigenvals.sum() > 0:
                planarity[i] = (eigenvals[0] - eigenvals[1]) / eigenvals.sum()

        return planarity

    def _calculate_wall_thickness(self, coordinates: torch.Tensor) -> float:
        """Calculate wall thickness."""
        if len(coordinates) < 3:
            return 0.0

        # Use PCA to find wall normal and thickness
        center = coordinates.mean(dim=0)
        relative_positions = coordinates - center

        # Compute covariance matrix
        cov_matrix = torch.matmul(relative_positions.T, relative_positions) / len(
            relative_positions
        )

        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)

        # Thickness is the smallest eigenvalue
        thickness = torch.sqrt(eigenvals[0]).item()

        return thickness

    def _cluster_voids(self, coordinates: torch.Tensor, scale: float) -> torch.Tensor:
        """Cluster void regions using distance-based clustering."""
        if len(coordinates) == 0:
            return torch.empty(0, dtype=torch.long, device=coordinates.device)

        # Use simple distance-based clustering
        edge_index = radius_graph(coordinates, r=scale * 2, loop=False)

        if edge_index.size(1) > 0:
            labels = self._connected_components(edge_index, len(coordinates))
        else:
            labels = torch.arange(len(coordinates), device=coordinates.device)

        return labels

    def _connected_components(
        self, edge_index: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        """Find connected components using Union-Find."""
        device = edge_index.device

        # Initialize each node as its own component
        parent = torch.arange(n_nodes, device=device)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union connected nodes
        for i in range(edge_index.size(1)):
            union(edge_index[0, i], edge_index[1, i])

        # Relabel components
        labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
        component_map = {}
        next_label = 0

        for i in range(n_nodes):
            root = find(i)
            if root not in component_map:
                component_map[root] = next_label
                next_label += 1
            labels[i] = component_map[root]

        return labels

    def _combine_structure_results(self, results: Dict) -> Dict:
        """Combine structure results from multiple scales."""
        combined = {
            "clusters": [],
            "voids": [],
            "walls": [],
            "statistics": {},
        }

        # Aggregate structures across scales
        for scale_name, scale_result in results.items():
            scale = scale_result["scale"]

            # Add clusters
            for cluster in scale_result["clusters"]:
                cluster["scale"] = scale
                combined["clusters"].append(cluster)

            # Add voids
            for void in scale_result["voids"]:
                void["scale"] = scale
                combined["voids"].append(void)

            # Add walls
            for wall in scale_result["walls"]:
                wall["scale"] = scale
                combined["walls"].append(wall)

        # Calculate statistics
        combined["statistics"] = {
            "n_clusters": len(combined["clusters"]),
            "n_voids": len(combined["voids"]),
            "n_walls": len(combined["walls"]),
            "total_structures": len(combined["clusters"])
            + len(combined["voids"])
            + len(combined["walls"]),
        }

        return combined


class CosmicWebAnalyzer:
    """
    Unified cosmic web analysis combining all structure types.

    Features:
    - Integrated cosmic web analysis
    - Multi-component detection
    - TensorDict integration
    - Comprehensive reporting
    """

    def __init__(
        self,
        device: str = None,
    ):
        """Initialize cosmic web analyzer."""
        if device is None:
            device = get_default_device()
        self.device = device
        self.filament_detector = FilamentDetector(device=device)
        self.structure_analyzer = StructureAnalyzer(device=device)

        logger.info(f"🌌 CosmicWebAnalyzer initialized on {self.device}")

    def analyze_cosmic_web(
        self,
        coordinates: Union[torch.Tensor, SpatialTensorDict],
        density_field: Optional[torch.Tensor] = None,
        scales: Optional[List[float]] = None,
    ) -> Dict:
        """
        Comprehensive cosmic web analysis.

        Args:
            coordinates: Object coordinates [N, 3]
            density_field: Optional density field [N]
            scales: Analysis scales

        Returns:
            Complete cosmic web analysis results
        """
        logger.info(
            f"🌌 Comprehensive cosmic web analysis: {coordinates.size(0) if hasattr(coordinates, 'size') else len(coordinates)} points"
        )

        # Analyze filaments
        filament_results = self.filament_detector.detect_filaments(
            coordinates, density_field, scales
        )

        # Analyze structures
        structure_results = self.structure_analyzer.analyze_structures(
            coordinates, density_field, scales
        )

        # Combine results
        combined = self._combine_cosmic_web_results(filament_results, structure_results)

        return {
            "filaments": filament_results,
            "structures": structure_results,
            "combined": combined,
            "coordinates": coordinates,
            "scales": scales,
        }

    def _combine_cosmic_web_results(
        self, filament_results: Dict, structure_results: Dict
    ) -> Dict:
        """Combine filament and structure analysis results."""
        combined = {
            "cosmic_web": {
                "filaments": filament_results["combined"]["filaments"],
                "clusters": structure_results["combined"]["clusters"],
                "voids": structure_results["combined"]["voids"],
                "walls": structure_results["combined"]["walls"],
            },
            "statistics": {
                "n_filaments": len(filament_results["combined"]["filaments"]),
                "n_clusters": len(structure_results["combined"]["clusters"]),
                "n_voids": len(structure_results["combined"]["voids"]),
                "n_walls": len(structure_results["combined"]["walls"]),
                "total_components": (
                    len(filament_results["combined"]["filaments"])
                    + len(structure_results["combined"]["clusters"])
                    + len(structure_results["combined"]["voids"])
                    + len(structure_results["combined"]["walls"])
                ),
            },
            "multi_scale": {
                "filaments": filament_results["multi_scale"],
                "structures": structure_results["multi_scale"],
            },
        }

        return combined
