"""
Cosmic Web Analyzer Component - Clean integration with AstroLab backend
Following Marimo 2025 reactive patterns
"""

from typing import Any, Dict, List, Optional, Tuple

import marimo as mo
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN

# Direct imports - clean
from astro_lab.ui.components.survey_manager import extract_coordinates


def create_analysis_config():
    """Create analysis configuration UI components"""

    scales_input = mo.ui.text(
        value="5,10,25,50",
        label="Clustering Scales (parsec, comma-separated)",
        placeholder="e.g., 5,10,25,50",
    )

    min_samples_input = mo.ui.slider(
        start=3, stop=20, value=5, label="Minimum Samples per Cluster"
    )

    return scales_input, min_samples_input


def parse_scales(scales_text: str) -> List[float]:
    """Parse scales from text input"""
    try:
        scales = [float(s.strip()) for s in scales_text.split(",") if s.strip()]
        return [s for s in scales if s > 0]  # Only positive scales
    except (ValueError, AttributeError):
        return [5.0, 10.0, 25.0, 50.0]  # Default scales


def run_cosmic_web_analysis(
    data: pl.DataFrame, scales: List[float], min_samples: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Run cosmic web analysis on data
    Returns analysis results or None on error
    """
    try:
        # Extract coordinates
        coordinates = extract_coordinates(data)
        if coordinates is None:
            return None

        # Run multi-scale clustering
        clustering_results = {}

        for scale in scales:
            # DBSCAN clustering at this scale
            dbscan = DBSCAN(eps=scale, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(coordinates)

            # Calculate statistics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            n_grouped = len(coordinates) - n_noise
            grouped_fraction = n_grouped / len(coordinates)

            # Cluster sizes
            cluster_sizes = [
                list(cluster_labels).count(cid)
                for cid in set(cluster_labels)
                if cid != -1
            ]

            clustering_results[scale] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_grouped": n_grouped,
                "grouped_fraction": grouped_fraction,
                "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "cluster_labels": cluster_labels.copy(),  # Store for visualization
            }

        return {
            "success": True,
            "n_objects": len(data),
            "coordinates": coordinates,
            "clustering_results": clustering_results,
            "scales": scales,
            "min_samples": min_samples,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def create_analysis_results_display(results: Optional[Dict[str, Any]]) -> mo.Html:
    """Create display for analysis results"""

    if results is None:
        return mo.md("⏳ **No analysis performed yet**")

    if not results.get("success", False):
        error_msg = results.get("error", "Unknown error")
        return mo.md(f"❌ **Analysis Error:** {error_msg}")

    # Success case - display results
    clustering_results = results.get("clustering_results", {})

    # Create results table
    results_list = []
    for scale, stats in clustering_results.items():
        results_list.append(
            {
                "Scale (pc)": scale,
                "Clusters": stats.get("n_clusters", 0),
                "Grouped": f"{stats.get('grouped_fraction', 0) * 100:.1f}%",
                "Largest Cluster": stats.get("largest_cluster_size", 0),
                "Noise Points": stats.get("n_noise", 0),
            }
        )

    df = pl.DataFrame(results_list)

    # Summary statistics
    n_objects = results.get("n_objects", 0)
    n_scales = len(clustering_results)
    avg_grouping = (
        np.mean(
            [stats.get("grouped_fraction", 0) for stats in clustering_results.values()]
        )
        * 100
    )
    total_clusters = sum(
        [stats.get("n_clusters", 0) for stats in clustering_results.values()]
    )

    summary_md = f"""
    ## ✅ Cosmic Web Analysis Complete
    
    **Objects Analyzed:** {n_objects:,}
    **Scales Analyzed:** {n_scales}
    **Average Grouping:** {avg_grouping:.1f}%
    **Total Clusters Found:** {total_clusters}
    
    💡 Higher grouping percentages indicate stronger cosmic structures.
    """

    return mo.vstack(
        [
            mo.md(summary_md),
            mo.ui.table(df.to_pandas(), label="Multi-Scale Clustering Results"),
        ]
    )


def create_scale_selector_from_results(
    results: Optional[Dict[str, Any]],
) -> Optional[mo.ui.dropdown]:
    """Create scale selector for visualization from analysis results"""

    if not results or not results.get("success", False):
        return None

    clustering_results = results.get("clustering_results", {})

    if not clustering_results:
        return None

    # Create options for each scale
    options = {}
    for scale, stats in clustering_results.items():
        n_clusters = stats.get("n_clusters", 0)
        grouped_pct = stats.get("grouped_fraction", 0) * 100
        options[f"{scale} pc ({n_clusters} clusters, {grouped_pct:.1f}% grouped)"] = (
            scale
        )

    return mo.ui.dropdown(
        options=options,
        label="Select Scale for Visualization",
        value=list(options.values())[0] if options else None,
    )


def get_cluster_data_for_scale(
    results: Dict[str, Any], scale: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get coordinates and cluster labels for a specific scale
    Returns (coordinates, cluster_labels) or None
    """
    if not results.get("success", False):
        return None

    clustering_results = results.get("clustering_results", {})
    scale_results = clustering_results.get(scale)

    if scale_results is None:
        return None

    coordinates = results.get("coordinates")
    cluster_labels = scale_results.get("cluster_labels")

    if coordinates is None or cluster_labels is None:
        return None

    return coordinates, cluster_labels


def create_analysis_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics from analysis results"""

    if not results.get("success", False):
        return {"error": "Analysis failed"}

    clustering_results = results.get("clustering_results", {})

    # Overall statistics
    scales = list(clustering_results.keys())
    grouping_rates = [
        stats.get("grouped_fraction", 0) for stats in clustering_results.values()
    ]
    cluster_counts = [
        stats.get("n_clusters", 0) for stats in clustering_results.values()
    ]

    return {
        "scales": scales,
        "grouping_rates": grouping_rates,
        "cluster_counts": cluster_counts,
        "best_scale": scales[np.argmax(grouping_rates)] if grouping_rates else None,
        "total_objects": results.get("n_objects", 0),
        "avg_grouping": np.mean(grouping_rates) if grouping_rates else 0,
    }


def create_analysis_button():
    """Create analysis trigger button"""
    return mo.ui.button("🔍 Run Cosmic Web Analysis", kind="success")


def create_analysis_status_message(is_running: bool = False) -> mo.Html:
    """Create status message for analysis"""
    if is_running:
        return mo.md("🔄 **Running cosmic web analysis...** This may take a moment.")
    else:
        return mo.md("✨ Ready to analyze cosmic web structures.")
