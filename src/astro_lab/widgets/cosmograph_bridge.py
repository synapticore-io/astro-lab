"""
Cosmograph Bridge - Enhanced API für UI Integration
"""

from typing import Any, Dict, List, Optional

import numpy as np

# Import echte Cosmograph Implementation + Enhanced Module
from .alcg.bridge import CosmographBridge as _CosmographBridge
from .enhanced import AstronomicalTensorBridge, ZeroCopyTensorConverter, to_cosmograph


class CosmographBridge:
    """
    Enhanced Cosmograph Bridge für UI Integration
    Nutzt die volle Power von AlcG + Enhanced Module für optimale Performance
    """

    def __init__(self):
        self._bridge = _CosmographBridge()
        self.tensor_bridge = AstronomicalTensorBridge()
        self.converter = ZeroCopyTensorConverter()

    def create_network_visualization(
        self,
        coordinates: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        node_size: float = 2.0,
        show_physics: bool = True,
        **kwargs,
    ):
        """
        Erstelle Enhanced Netzwerk Visualisierung

        Args:
            coordinates: 3D Koordinaten Array (N, 3)
            cluster_labels: Optional cluster labels
            node_size: Node Größe
            show_physics: Physik Simulation aktivieren
            **kwargs: Zusätzliche Parameter

        Returns:
            Enhanced Cosmograph Widget
        """

        # Enhanced Tensor Conversion
        cosmograph_data = to_cosmograph(
            coordinates, cluster_labels=cluster_labels, **kwargs
        )

        # Erstelle Enhanced Node Daten
        nodes = []
        for i, coord in enumerate(coordinates):
            node = {
                "id": str(i),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]) if len(coord) > 2 else 0.0,
                "size": node_size * self._get_enhanced_size_factor(i, cluster_labels),
                "color": self._get_enhanced_node_color(i, cluster_labels),
                "opacity": self._get_enhanced_opacity(i, cluster_labels),
            }
            nodes.append(node)

        # Enhanced Edge Creation
        edges = self._create_enhanced_edges(coordinates, cluster_labels, **kwargs)

        # Verwende echte Bridge mit Enhanced Features
        return self._bridge.create_visualization(
            nodes=nodes,
            edges=edges,
            show_physics=show_physics,
            enhanced_features=True,
            **kwargs,
        )

    def create_cosmic_web_network(
        self, coordinates: np.ndarray, scale: float, n_clusters: int, **kwargs
    ):
        """
        Erstelle Enhanced Cosmic Web Netzwerk

        Args:
            coordinates: 3D Koordinaten
            scale: Analyse Skala
            n_clusters: Anzahl gefundener Cluster
            **kwargs: Zusätzliche Parameter

        Returns:
            Enhanced Cosmograph Widget
        """

        # Enhanced Cluster Analysis
        cluster_labels = self._create_enhanced_clusters(coordinates, n_clusters, scale)

        # Enhanced Physics Parameters basierend auf Skala
        physics_config = self._get_enhanced_physics_config(scale, n_clusters)

        return self.create_network_visualization(
            coordinates=coordinates,
            cluster_labels=cluster_labels,
            node_size=2.0 * (scale / 10.0),  # Scale-abhängige Größe
            show_physics=True,
            physics_config=physics_config,
            **kwargs,
        )

    def from_cosmic_web_results(
        self, analysis_results: Dict[str, Any], survey_name: str = "unknown", **kwargs
    ):
        """
        Erstelle Enhanced Visualisierung aus Cosmic Web Analyse Ergebnissen

        Args:
            analysis_results: Ergebnisse der Cosmic Web Analyse
            survey_name: Name des Surveys
            **kwargs: Zusätzliche Parameter

        Returns:
            Enhanced Cosmograph Widget
        """

        coordinates = analysis_results.get("coordinates")
        if coordinates is None:
            raise ValueError("Keine Koordinaten in analysis_results gefunden")

        # Enhanced Analysis Processing
        clustering_results = analysis_results.get("clustering_results", {})
        if clustering_results:
            # Enhanced Multi-Scale Analysis
            best_scale, best_stats = self._analyze_enhanced_clustering(
                clustering_results
            )
            n_clusters = best_stats.get("n_clusters", 0)
        else:
            best_scale = 10.0
            n_clusters = 5

        # Enhanced Visualization mit Survey-spezifischen Features
        enhanced_kwargs = self._get_survey_enhancement(survey_name)
        enhanced_kwargs.update(kwargs)

        return self.create_cosmic_web_network(
            coordinates=coordinates,
            scale=float(str(best_scale).replace("pc", "").replace("Mpc", "")),
            n_clusters=n_clusters,
            survey_type=survey_name,
            **enhanced_kwargs,
        )

    def _get_enhanced_node_color(
        self, node_idx: int, cluster_labels: Optional[np.ndarray]
    ) -> str:
        """Enhanced Farb-Schema basierend auf Cluster Labels"""

        # Enhanced Color Palette für bessere Unterscheidung
        colors = [
            "#FFD700",  # Gold - Cluster 0
            "#FF6B6B",  # Coral Red - Cluster 1
            "#4ECDC4",  # Teal - Cluster 2
            "#45B7D1",  # Sky Blue - Cluster 3
            "#96CEB4",  # Mint Green - Cluster 4
            "#FFEAA7",  # Warm Yellow - Cluster 5
            "#DDA0DD",  # Plum - Cluster 6
            "#FFA07A",  # Light Salmon - Cluster 7
            "#98D8C8",  # Mint - Cluster 8
            "#F7DC6F",  # Light Gold - Cluster 9
            "#BB8FCE",  # Medium Orchid - Cluster 10
            "#85C1E9",  # Light Blue - Cluster 11
        ]

        if cluster_labels is None:
            return colors[0]  # Default Gold

        label = cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1

        if label == -1:  # Enhanced Noise Visualization
            return "#404040"  # Dark Gray für Noise

        return colors[label % len(colors)]

    def _get_enhanced_size_factor(
        self, node_idx: int, cluster_labels: Optional[np.ndarray]
    ) -> float:
        """Enhanced Size Factor basierend auf Cluster Membership"""

        if cluster_labels is None:
            return 1.0

        label = cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1

        if label == -1:  # Noise nodes kleiner
            return 0.7
        else:  # Cluster nodes größer
            return 1.2

    def _get_enhanced_opacity(
        self, node_idx: int, cluster_labels: Optional[np.ndarray]
    ) -> float:
        """Enhanced Opacity für bessere Visualisierung"""

        if cluster_labels is None:
            return 0.8

        label = cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1

        if label == -1:  # Noise nodes transparenter
            return 0.4
        else:  # Cluster nodes opaker
            return 0.9

    def _create_enhanced_edges(
        self, coordinates: np.ndarray, cluster_labels: Optional[np.ndarray], **kwargs
    ) -> List[Dict]:
        """Erstelle Enhanced Edges mit intelligentem Routing"""

        from sklearn.neighbors import NearestNeighbors

        k = kwargs.get("k_neighbors", 5)

        # Enhanced k-NN mit Cluster-aware Weighting
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

        edges = []
        for i, (neighbor_distances, neighbors) in enumerate(zip(distances, indices)):
            for j, neighbor_idx in enumerate(neighbors[1:]):  # Skip self
                # Enhanced Edge Weighting
                distance = neighbor_distances[j + 1]

                # Cluster-aware edge strength
                edge_strength = self._calculate_enhanced_edge_strength(
                    i, neighbor_idx, distance, cluster_labels
                )

                edge = {
                    "source": str(i),
                    "target": str(neighbor_idx),
                    "weight": edge_strength,
                    "distance": float(distance),
                    "enhanced": True,
                }
                edges.append(edge)

        return edges

    def _calculate_enhanced_edge_strength(
        self,
        node1: int,
        node2: int,
        distance: float,
        cluster_labels: Optional[np.ndarray],
    ) -> float:
        """Berechne Enhanced Edge Strength"""

        base_strength = 1.0 / (1.0 + distance)  # Distance-based base strength

        if cluster_labels is None:
            return base_strength

        label1 = cluster_labels[node1] if node1 < len(cluster_labels) else -1
        label2 = cluster_labels[node2] if node2 < len(cluster_labels) else -1

        # Enhanced: Same cluster edges stronger
        if label1 == label2 and label1 != -1:
            return base_strength * 2.0  # Intra-cluster edges stronger
        elif label1 != -1 and label2 != -1:
            return base_strength * 0.5  # Inter-cluster edges weaker
        else:
            return base_strength * 0.3  # Noise edges weakest

    def _create_enhanced_clusters(
        self, coordinates: np.ndarray, n_clusters: int, scale: float
    ) -> np.ndarray:
        """Enhanced Cluster Creation mit Multi-Algorithm Approach"""

        if n_clusters <= 1:
            return np.zeros(len(coordinates), dtype=int)

        # Enhanced: Verwende DBSCAN für density-based clustering
        from sklearn.cluster import DBSCAN

        # Scale-adaptive epsilon
        eps = scale * 0.5  # Adaptive epsilon basierend auf Skala
        min_samples = max(3, int(n_clusters * 0.1))  # Adaptive min_samples

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coordinates)

        return labels

    def _get_enhanced_physics_config(
        self, scale: float, n_clusters: int
    ) -> Dict[str, Any]:
        """Enhanced Physics Configuration"""

        return {
            "gravity": 0.1 * (scale / 10.0),  # Scale-dependent gravity
            "repulsion": 1.0 + (n_clusters / 10.0),  # Cluster-dependent repulsion
            "damping": 0.8,
            "enhanced_simulation": True,
        }

    def _analyze_enhanced_clustering(self, clustering_results: Dict[str, Any]) -> tuple:
        """Enhanced Clustering Analysis"""

        # Finde beste Skala basierend auf Enhanced Metriken
        best_score = 0
        best_scale = None
        best_stats = {}

        for scale, stats in clustering_results.items():
            # Enhanced Scoring: Kombiniere mehrere Metriken
            grouped_fraction = stats.get("grouped_fraction", 0)
            n_clusters = stats.get("n_clusters", 0)

            # Enhanced Score: Balance zwischen Gruppierung und Cluster-Anzahl
            score = grouped_fraction * 0.7 + min(n_clusters / 10.0, 1.0) * 0.3

            if score > best_score:
                best_score = score
                best_scale = scale
                best_stats = stats

        return best_scale or 10.0, best_stats

    def _get_survey_enhancement(self, survey_name: str) -> Dict[str, Any]:
        """Survey-spezifische Enhanced Features"""

        enhancements = {
            "gaia": {
                "node_color_scheme": "stellar",
                "physics_mode": "galactic",
                "edge_weighting": "proper_motion",
            },
            "sdss": {
                "node_color_scheme": "redshift",
                "physics_mode": "cosmological",
                "edge_weighting": "distance",
            },
            "nsa": {
                "node_color_scheme": "morphology",
                "physics_mode": "cluster",
                "edge_weighting": "luminosity",
            },
        }

        return enhancements.get(
            survey_name,
            {
                "node_color_scheme": "default",
                "physics_mode": "standard",
                "edge_weighting": "distance",
            },
        )


# Export für UI
__all__ = ["CosmographBridge"]
