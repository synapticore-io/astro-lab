#!/usr/bin/env python3
"""
Generate Visual Outputs for AstroLab Catalog
============================================

Creates publication-quality visualizations from the AstroLab catalog:
- 3D interactive cosmic web plots (Plotly HTML)
- Structure classification plots
- Multi-scale comparison plots
- Statistical distribution plots

Usage:
    python scripts/generate_visualizations.py [--catalog PATH] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_catalog(catalog_path: Path) -> pl.DataFrame:
    """Load the AstroLab catalog."""
    logger.info(f"Loading catalog from {catalog_path}")
    df = pl.read_parquet(catalog_path)
    logger.info(f"   ✓ Loaded {len(df):,} sources with {len(df.columns)} columns")
    return df


def create_3d_cosmic_web_plot(
    df: pl.DataFrame,
    output_path: Path,
    scale: float = 10.0,
    max_points: int = 10000
) -> None:
    """
    Create interactive 3D cosmic web visualization.
    
    Args:
        df: Catalog dataframe
        output_path: Path to save HTML file
        scale: Clustering scale to visualize
        max_points: Maximum number of points to plot
    """
    logger.info(f"Creating 3D cosmic web plot at scale {scale} pc")
    
    # Find coordinate columns
    coord_cols = []
    for col in ['x', 'y', 'z']:
        if col in df.columns:
            coord_cols.append(col)
        else:
            # Try survey-prefixed columns
            for prefix in ['gaia', 'sdss', 'twomass']:
                prefixed_col = f"{prefix}_{col}"
                if prefixed_col in df.columns:
                    coord_cols.append(prefixed_col)
                    break
    
    if len(coord_cols) != 3:
        logger.error(f"Could not find 3D coordinates. Available columns: {df.columns}")
        return
    
    # Sample if needed
    if len(df) > max_points:
        logger.info(f"   Sampling {max_points:,} of {len(df):,} points")
        df = df.sample(n=max_points, shuffle=True)
    
    # Extract coordinates
    x = df[coord_cols[0]].to_numpy()
    y = df[coord_cols[1]].to_numpy()
    z = df[coord_cols[2]].to_numpy()
    
    # Get cosmic web classification
    class_col = f"cosmic_web_class_{scale}pc"
    if class_col in df.columns:
        structure_class = df[class_col].to_numpy()
        
        # Map classes to names
        class_names = {
            0: "Field",
            1: "Filament", 
            2: "Void",
            3: "Node"
        }
        class_labels = [class_names.get(int(c), f"Class {c}") for c in structure_class]
        
        # Create color scale
        color_map = {
            "Field": "#1f77b4",    # Blue
            "Filament": "#ff7f0e", # Orange
            "Void": "#2ca02c",     # Green
            "Node": "#d62728"      # Red
        }
        colors = [color_map.get(label, "#gray") for label in class_labels]
    else:
        logger.warning(f"   ⚠ Column {class_col} not found, using uniform color")
        class_labels = ["Source"] * len(df)
        colors = ["#1f77b4"] * len(df)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.6,
            line=dict(width=0)
        ),
        text=class_labels,
        hovertemplate='<b>%{text}</b><br>' +
                     'X: %{x:.1f} pc<br>' +
                     'Y: %{y:.1f} pc<br>' +
                     'Z: %{z:.1f} pc<br>' +
                     '<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title=f"AstroLab Cosmic Web Structure (Scale: {scale} pc)",
        scene=dict(
            xaxis_title="X (parsec)",
            yaxis_title="Y (parsec)",
            zaxis_title="Z (parsec)",
            bgcolor="black",
            xaxis=dict(backgroundcolor="black", gridcolor="gray"),
            yaxis=dict(backgroundcolor="black", gridcolor="gray"),
            zaxis=dict(backgroundcolor="black", gridcolor="gray"),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        showlegend=False,
        height=800
    )
    
    # Save
    fig.write_html(output_path)
    logger.info(f"   ✓ Saved to {output_path}")


def create_structure_distribution_plot(
    df: pl.DataFrame,
    output_path: Path,
    scales: list = None
) -> None:
    """Create bar plot showing structure type distribution across scales."""
    logger.info("Creating structure distribution plot")
    
    if scales is None:
        # Auto-detect scales from columns
        scales = []
        for col in df.columns:
            if col.startswith("cosmic_web_class_") and col.endswith("pc"):
                scale_str = col.replace("cosmic_web_class_", "").replace("pc", "")
                try:
                    scales.append(float(scale_str))
                except ValueError:
                    pass
        scales.sort()
    
    if not scales:
        logger.warning("   ⚠ No cosmic web classification columns found")
        return
    
    # Collect data for each scale
    data = []
    class_names = ["Field", "Filament", "Void", "Node"]
    
    for scale in scales:
        class_col = f"cosmic_web_class_{scale}pc"
        if class_col not in df.columns:
            continue
        
        counts = df[class_col].value_counts().sort("cosmic_web_class_" + f"{scale}pc")
        for row in counts.iter_rows():
            class_idx = int(row[0])
            count = row[1]
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            data.append({
                "Scale (pc)": f"{scale}",
                "Structure Type": class_name,
                "Count": count
            })
    
    # Create DataFrame for plotting
    plot_df = pl.DataFrame(data)
    
    # Create grouped bar chart
    fig = px.bar(
        plot_df.to_pandas(),
        x="Scale (pc)",
        y="Count",
        color="Structure Type",
        barmode="group",
        title="Cosmic Web Structure Distribution Across Scales",
        color_discrete_map={
            "Field": "#1f77b4",
            "Filament": "#ff7f0e",
            "Void": "#2ca02c",
            "Node": "#d62728"
        }
    )
    
    fig.update_layout(
        xaxis_title="Clustering Scale",
        yaxis_title="Number of Sources",
        font=dict(size=12),
        height=500
    )
    
    fig.write_html(output_path)
    logger.info(f"   ✓ Saved to {output_path}")


def create_density_distribution_plot(
    df: pl.DataFrame,
    output_path: Path,
    scale: float = 10.0
) -> None:
    """Create density distribution histogram."""
    logger.info(f"Creating density distribution plot at scale {scale} pc")
    
    density_col = f"density_{scale}pc"
    if density_col not in df.columns:
        logger.warning(f"   ⚠ Column {density_col} not found")
        return
    
    density = df[density_col].to_numpy()
    
    fig = go.Figure(data=[go.Histogram(
        x=density,
        nbinsx=50,
        marker_color='#1f77b4',
        opacity=0.7
    )])
    
    fig.update_layout(
        title=f"Local Density Distribution (Scale: {scale} pc)",
        xaxis_title="Density",
        yaxis_title="Count",
        font=dict(size=12),
        height=400
    )
    
    fig.write_html(output_path)
    logger.info(f"   ✓ Saved to {output_path}")


def create_multi_scale_comparison(
    df: pl.DataFrame,
    output_path: Path,
    sample_size: int = 5000
) -> None:
    """Create multi-panel comparison of structures at different scales."""
    logger.info("Creating multi-scale comparison plot")
    
    # Detect available scales
    scales = []
    for col in df.columns:
        if col.startswith("cosmic_web_class_") and col.endswith("pc"):
            scale_str = col.replace("cosmic_web_class_", "").replace("pc", "")
            try:
                scales.append(float(scale_str))
            except ValueError:
                pass
    scales.sort()
    
    if len(scales) < 2:
        logger.warning("   ⚠ Need at least 2 scales for comparison")
        return
    
    # Sample data
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, shuffle=True)
    else:
        df_sample = df
    
    # Find coordinate columns
    coord_cols = []
    for col in ['x', 'y', 'z']:
        if col in df_sample.columns:
            coord_cols.append(col)
        else:
            for prefix in ['gaia', 'sdss', 'twomass']:
                prefixed_col = f"{prefix}_{col}"
                if prefixed_col in df_sample.columns:
                    coord_cols.append(prefixed_col)
                    break
    
    if len(coord_cols) < 3:
        logger.warning("   ⚠ Could not find 3D coordinates")
        return
    
    x = df_sample[coord_cols[0]].to_numpy()
    y = df_sample[coord_cols[1]].to_numpy()
    
    # Create subplots (use first 4 scales)
    n_scales = min(4, len(scales))
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Scale: {scales[i]} pc" for i in range(n_scales)],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    class_names = {0: "Field", 1: "Filament", 2: "Void", 3: "Node"}
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    
    for idx, scale in enumerate(scales[:n_scales]):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        class_col = f"cosmic_web_class_{scale}pc"
        if class_col in df_sample.columns:
            classes = df_sample[class_col].to_numpy()
            
            # Plot each class separately for legend
            for class_id in sorted(set(classes)):
                mask = classes == class_id
                fig.add_trace(
                    go.Scatter(
                        x=x[mask],
                        y=y[mask],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=colors.get(class_id, "#gray"),
                            opacity=0.6
                        ),
                        name=class_names.get(class_id, f"Class {class_id}"),
                        showlegend=(idx == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
    
    fig.update_xaxes(title_text="X (pc)")
    fig.update_yaxes(title_text="Y (pc)")
    fig.update_layout(
        title_text="Multi-Scale Cosmic Web Structure Comparison",
        height=800,
        showlegend=True
    )
    
    fig.write_html(output_path)
    logger.info(f"   ✓ Saved to {output_path}")


def generate_all_visualizations(
    catalog_path: Path,
    output_dir: Path = Path("data/visualizations")
) -> None:
    """Generate all visualizations from catalog."""
    logger.info("=" * 80)
    logger.info("🎨 AstroLab Visualization Generation")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    df = load_catalog(catalog_path)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations:")
    logger.info("-" * 80)
    
    # 3D cosmic web plot
    create_3d_cosmic_web_plot(
        df,
        output_dir / "cosmic_web_3d.html",
        scale=10.0,
        max_points=10000
    )
    
    # Structure distribution
    create_structure_distribution_plot(
        df,
        output_dir / "structure_distribution.html"
    )
    
    # Density distribution
    create_density_distribution_plot(
        df,
        output_dir / "density_distribution.html",
        scale=10.0
    )
    
    # Multi-scale comparison
    create_multi_scale_comparison(
        df,
        output_dir / "multi_scale_comparison.html",
        sample_size=5000
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Visualization generation complete!")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("\nGenerated files:")
    for file in sorted(output_dir.glob("*.html")):
        logger.info(f"  - {file.name}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from AstroLab catalog"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalogs/astrolab_catalog_v1.parquet"),
        help="Path to catalog file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visualizations"),
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    if not args.catalog.exists():
        logger.error(f"❌ Catalog not found: {args.catalog}")
        logger.info("Run scripts/generate_astrolab_catalog.py first to create the catalog")
        return 1
    
    try:
        generate_all_visualizations(args.catalog, args.output_dir)
        print(f"\n✅ Success! Open the HTML files in {args.output_dir} to view visualizations")
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
