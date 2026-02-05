"""
Plotly Bridge - Enhanced API für UI Integration
"""

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from .enhanced import AstronomicalTensorBridge, ImageProcessor, to_plotly

# Import echte Plotly Implementation + Enhanced Module
from .plotly.bridge import AstronomicalPlotlyBridge


class EnhancedPlotlyBridge:
    """Enhanced Plotly Bridge mit verbesserter Performance und Features"""

    def __init__(self):
        self.bridge = AstronomicalPlotlyBridge()
        self.tensor_bridge = AstronomicalTensorBridge()
        self.image_processor = ImageProcessor()


def create_3d_scatter_plot(
    coordinates: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "3D Scatter Plot",
    point_size: int = 3,
    **kwargs,
) -> go.Figure:
    """
    Enhanced 3D Scatter Plot - UI-freundliche API mit Enhanced Features

    Args:
        coordinates: 3D Koordinaten Array (N, 3)
        cluster_labels: Optional cluster labels für Enhanced Farbkodierung
        title: Plot Titel
        point_size: Punkt Größe
        **kwargs: Zusätzliche Enhanced Parameter

    Returns:
        Enhanced Plotly Figure
    """

    # Enhanced Tensor Conversion
    enhanced_bridge = EnhancedPlotlyBridge()
    plotly_data = to_plotly(coordinates, cluster_labels=cluster_labels, **kwargs)

    # Enhanced Color Scheme
    if cluster_labels is not None:
        colors = cluster_labels
        colorscale = kwargs.get("colorscale", "Viridis")

        # Enhanced: Custom colorscale für Cluster
        if kwargs.get("enhanced_colors", True):
            colorscale = [
                [0.0, "#404040"],  # Noise (grau)
                [0.2, "#FFD700"],  # Cluster 0 (gold)
                [0.4, "#FF6B6B"],  # Cluster 1 (rot)
                [0.6, "#4ECDC4"],  # Cluster 2 (türkis)
                [0.8, "#45B7D1"],  # Cluster 3 (blau)
                [1.0, "#96CEB4"],  # Cluster 4 (grün)
            ]
    else:
        # Enhanced: Distance-based coloring
        distances = np.linalg.norm(coordinates, axis=1)
        colors = distances
        colorscale = "Viridis"

    # Enhanced Figure Creation
    fig = go.Figure()

    # Enhanced Scatter mit optimierter Performance
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=colors,
                colorscale=colorscale,
                opacity=kwargs.get("opacity", 0.8),
                line=dict(width=0.5, color="DarkSlateGrey")
                if kwargs.get("enhanced_borders", False)
                else None,
                colorbar=dict(
                    title="Cluster" if cluster_labels is not None else "Distance"
                ),
                sizemode="diameter",
            ),
            name=kwargs.get("name", "Objects"),
            hovertemplate="<b>%{text}</b><br>"
            + "X: %{x:.2f}<br>"
            + "Y: %{y:.2f}<br>"
            + "Z: %{z:.2f}<br>"
            + "<extra></extra>",
            text=[f"Object {i}" for i in range(len(coordinates))],
        )
    )

    # Enhanced Layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.6), projection=dict(type="perspective")
            ),
            bgcolor="rgba(0,0,0,0.1)"
            if kwargs.get("enhanced_background", True)
            else "white",
        ),
        template="plotly_dark" if kwargs.get("dark_theme", True) else "plotly_white",
        width=kwargs.get("width", 800),
        height=kwargs.get("height", 600),
    )

    return fig


def create_cosmic_web_plot(
    coordinates: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    edges: Optional[np.ndarray] = None,
    title: str = "Cosmic Web Structure",
    **kwargs,
) -> go.Figure:
    """
    Enhanced Cosmic Web Plot mit Clustern und Filaments

    Args:
        coordinates: 3D Koordinaten Array (N, 3)
        cluster_labels: Cluster Labels für Enhanced Farbkodierung
        edges: Edge connections für Enhanced Filaments
        title: Plot Titel
        **kwargs: Enhanced Parameter

    Returns:
        Enhanced Plotly Figure
    """

    # Basis 3D Scatter
    fig = create_3d_scatter_plot(
        coordinates=coordinates,
        cluster_labels=cluster_labels,
        title=title,
        point_size=kwargs.get("point_size", 2),
        enhanced_colors=True,
        **kwargs,
    )

    # Enhanced Filament Visualization
    if edges is not None:
        max_edges = kwargs.get("max_edges", 1000)  # Performance limit
        edge_opacity = kwargs.get("edge_opacity", 0.3)

        for i, edge in enumerate(edges[:max_edges]):
            if len(edge) >= 2:
                idx1, idx2 = edge[0], edge[1]

                # Enhanced: Color edges by cluster similarity
                edge_color = "rgba(100,100,100,0.3)"
                if cluster_labels is not None:
                    if (
                        cluster_labels[idx1] == cluster_labels[idx2]
                        and cluster_labels[idx1] != -1
                    ):
                        edge_color = (
                            f"rgba(255,215,0,{edge_opacity})"  # Same cluster: gold
                        )
                    else:
                        edge_color = f"rgba(128,128,128,{edge_opacity * 0.5})"  # Different clusters: dim

                fig.add_trace(
                    go.Scatter3d(
                        x=[coordinates[idx1, 0], coordinates[idx2, 0], None],
                        y=[coordinates[idx1, 1], coordinates[idx2, 1], None],
                        z=[coordinates[idx1, 2], coordinates[idx2, 2], None],
                        mode="lines",
                        line=dict(color=edge_color, width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Enhanced: Auto-generate k-NN edges if none provided
    elif kwargs.get("auto_edges", False):
        from sklearn.neighbors import NearestNeighbors

        k = kwargs.get("k_neighbors", 5)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

        for i, neighbors in enumerate(indices):
            for j in neighbors[1 : k + 1]:  # Enhanced: limit edges per node
                edge_color = f"rgba(100,100,100,{kwargs.get('auto_edge_opacity', 0.2)})"

                fig.add_trace(
                    go.Scatter3d(
                        x=[coordinates[i, 0], coordinates[j, 0], None],
                        y=[coordinates[i, 1], coordinates[j, 1], None],
                        z=[coordinates[i, 2], coordinates[j, 2], None],
                        mode="lines",
                        line=dict(color=edge_color, width=0.5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    return fig


def create_survey_comparison(
    survey_data: Dict[str, np.ndarray],
    title: str = "Enhanced Survey Comparison",
    **kwargs,
) -> go.Figure:
    """
    Enhanced Survey Comparison mit verbesserter Visualisierung

    Args:
        survey_data: Dict mit survey_name -> coordinates
        title: Plot Titel
        **kwargs: Enhanced Parameter

    Returns:
        Enhanced Plotly Figure
    """

    fig = go.Figure()

    # Enhanced Color Palette für Survey Comparison
    enhanced_colors = [
        "#FFD700",  # Gaia - Gold
        "#4169E1",  # SDSS - Royal Blue
        "#FF6347",  # NSA - Tomato
        "#32CD32",  # TNG50 - Lime Green
        "#FF69B4",  # Exoplanet - Hot Pink
        "#20B2AA",  # WISE - Light Sea Green
        "#DDA0DD",  # LINEAR - Plum
        "#F0E68C",  # RR Lyrae - Khaki
    ]

    for i, (survey_name, coordinates) in enumerate(survey_data.items()):
        color = enhanced_colors[i % len(enhanced_colors)]

        # Enhanced: Survey-specific point sizes
        point_size = kwargs.get("point_size", 3)
        if "gaia" in survey_name.lower():
            point_size *= 0.8  # Smaller for star data
        elif "galaxy" in survey_name.lower() or "nsa" in survey_name.lower():
            point_size *= 1.5  # Larger for galaxy data

        fig.add_trace(
            go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=kwargs.get("opacity", 0.7),
                    line=dict(width=0.5, color="white")
                    if kwargs.get("enhanced_borders", True)
                    else None,
                ),
                name=f"{survey_name.upper()} ({len(coordinates):,} objects)",
                hovertemplate=f"<b>{survey_name.upper()}</b><br>"
                + "X: %{x:.2f}<br>"
                + "Y: %{y:.2f}<br>"
                + "Z: %{z:.2f}<br>"
                + "<extra></extra>",
            )
        )

    # Enhanced Layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        scene=dict(
            xaxis_title="X (spatial unit)",
            yaxis_title="Y (spatial unit)",
            zaxis_title="Z (spatial unit)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="rgba(0,0,0,0.05)",
        ),
        template="plotly_dark" if kwargs.get("dark_theme", True) else "plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=kwargs.get("width", 900),
        height=kwargs.get("height", 700),
    )

    return fig


def create_hr_diagram(
    magnitudes: np.ndarray, bands: list, title: str = "Enhanced HR Diagram", **kwargs
) -> go.Figure:
    """
    Enhanced Hertzsprung-Russell Diagram

    Args:
        magnitudes: Magnitude Array (N, n_bands)
        bands: Band Namen
        title: Plot Titel
        **kwargs: Enhanced Parameter

    Returns:
        Enhanced Plotly Figure
    """

    enhanced_bridge = EnhancedPlotlyBridge()

    # Enhanced HR Diagram Creation
    fig = enhanced_bridge.bridge._create_hr_diagram(magnitudes, bands, **kwargs)

    # Enhanced Styling
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        template="plotly_dark" if kwargs.get("dark_theme", True) else "plotly_white",
        width=kwargs.get("width", 800),
        height=kwargs.get("height", 600),
    )

    # Enhanced: Add stellar evolution tracks if requested
    if kwargs.get("show_evolution_tracks", False):
        # Placeholder für enhanced stellar evolution tracks
        pass

    return fig


# Export für UI
__all__ = [
    "create_3d_scatter_plot",
    "create_cosmic_web_plot",
    "create_survey_comparison",
    "create_hr_diagram",
]
