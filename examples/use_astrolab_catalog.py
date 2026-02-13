"""
Example: Using the AstroLab Consolidated Catalog
================================================

This example demonstrates how to work with the consolidated AstroLab catalog
that combines multiple surveys with cosmic web features.

Features demonstrated:
1. Loading and exploring the catalog
2. Filtering by cosmic web structure type
3. Multi-scale analysis
4. Creating custom visualizations
5. Extracting subsets for specific science cases

Prerequisites:
    Generate the catalog first:
    $ python scripts/generate_astrolab_catalog.py --max-samples 10000
"""

import logging
from pathlib import Path

import polars as pl
import torch

# Optional visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available - install with: uv pip install plotly")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_catalog(catalog_path: Path = Path("data/catalogs/astrolab_catalog_v1.parquet")):
    """Load the AstroLab catalog."""
    if not catalog_path.exists():
        print(f"❌ Catalog not found: {catalog_path}")
        print("\nGenerate it first with:")
        print("  python scripts/generate_astrolab_catalog.py --max-samples 10000")
        return None
    
    logger.info(f"Loading catalog from {catalog_path}")
    df = pl.read_parquet(catalog_path)
    logger.info(f"✓ Loaded {len(df):,} sources with {len(df.columns)} columns")
    return df


def explore_catalog_structure(df: pl.DataFrame):
    """Print catalog structure and statistics."""
    print("\n" + "=" * 80)
    print("CATALOG STRUCTURE")
    print("=" * 80)
    
    print(f"\nTotal sources: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Show column groups
    print("\nColumn groups:")
    
    # Coordinate columns
    coord_cols = [c for c in df.columns if c in ['x', 'y', 'z', 'ra', 'dec', 'distance_pc']
                  or c.endswith('_x') or c.endswith('_y') or c.endswith('_z')]
    if coord_cols:
        print(f"  Coordinates: {', '.join(coord_cols)}")
    
    # Cosmic web columns
    cw_cols = [c for c in df.columns if 'cosmic_web_class' in c or 'density' in c or 'anisotropy' in c]
    if cw_cols:
        print(f"  Cosmic web: {len(cw_cols)} columns")
        for col in sorted(cw_cols):
            print(f"    - {col}")
    
    # Photometry columns
    phot_cols = [c for c in df.columns if 'mag' in c.lower() or any(band in c for band in ['g', 'r', 'i', 'z', 'u', 'j', 'h', 'k', 'bp', 'rp'])]
    if phot_cols:
        print(f"  Photometry: {len(phot_cols)} columns")
    
    # Metadata
    meta_cols = [c for c in df.columns if c in ['catalog_version', 'processing_date']]
    if meta_cols:
        print(f"  Metadata: {', '.join(meta_cols)}")


def analyze_cosmic_web_distribution(df: pl.DataFrame, scale: float = 10.0):
    """Analyze distribution of cosmic web structures."""
    print("\n" + "=" * 80)
    print(f"COSMIC WEB DISTRIBUTION (Scale: {scale} pc)")
    print("=" * 80)
    
    class_col = f"cosmic_web_class_{scale}pc"
    if class_col not in df.columns:
        print(f"⚠️  Column {class_col} not found")
        return
    
    class_names = {0: "Field", 1: "Filament", 2: "Void", 3: "Node"}
    counts = df[class_col].value_counts().sort(class_col)
    
    total = len(df)
    print("\nStructure type distribution:")
    for row in counts.iter_rows():
        class_id = int(row[0])
        count = row[1]
        percentage = (count / total) * 100
        class_name = class_names.get(class_id, f"Class {class_id}")
        print(f"  {class_name:12s}: {count:8,} ({percentage:5.1f}%)")
    
    return counts


def filter_by_structure(df: pl.DataFrame, structure_type: str, scale: float = 10.0):
    """Filter catalog by cosmic web structure type."""
    class_col = f"cosmic_web_class_{scale}pc"
    if class_col not in df.columns:
        print(f"⚠️  Column {class_col} not found")
        return None
    
    structure_map = {"field": 0, "filament": 1, "void": 2, "node": 3}
    class_id = structure_map.get(structure_type.lower())
    
    if class_id is None:
        print(f"⚠️  Unknown structure type: {structure_type}")
        print(f"    Valid types: {', '.join(structure_map.keys())}")
        return None
    
    filtered = df.filter(pl.col(class_col) == class_id)
    print(f"\n✓ Filtered to {len(filtered):,} {structure_type} sources (scale: {scale} pc)")
    return filtered


def compare_scales(df: pl.DataFrame):
    """Compare cosmic web classification across different scales."""
    print("\n" + "=" * 80)
    print("MULTI-SCALE COMPARISON")
    print("=" * 80)
    
    # Find available scales
    scales = []
    for col in df.columns:
        if col.startswith("cosmic_web_class_") and col.endswith("pc"):
            scale_str = col.replace("cosmic_web_class_", "").replace("pc", "")
            try:
                scales.append(float(scale_str))
            except ValueError:
                pass
    
    if not scales:
        print("⚠️  No cosmic web classification columns found")
        return
    
    scales.sort()
    print(f"\nAvailable scales: {scales}")
    
    class_names = {0: "Field", 1: "Filament", 2: "Void", 3: "Node"}
    
    print("\nStructure distribution across scales:")
    print(f"{'Scale (pc)':<12} {'Field':<10} {'Filament':<10} {'Void':<10} {'Node':<10}")
    print("-" * 52)
    
    for scale in scales:
        class_col = f"cosmic_web_class_{scale}pc"
        counts = df[class_col].value_counts().sort(class_col)
        
        dist = {int(row[0]): row[1] for row in counts.iter_rows()}
        total = len(df)
        
        scale_str = f"{scale:<12.1f}"
        field_pct = f"{dist.get(0, 0) / total * 100:>5.1f}%"
        filament_pct = f"{dist.get(1, 0) / total * 100:>5.1f}%"
        void_pct = f"{dist.get(2, 0) / total * 100:>5.1f}%"
        node_pct = f"{dist.get(3, 0) / total * 100:>5.1f}%"
        
        print(f"{scale_str} {field_pct:<10} {filament_pct:<10} {void_pct:<10} {node_pct:<10}")


def visualize_structure_subset(df: pl.DataFrame, structure_type: str = "filament", 
                               scale: float = 10.0, max_points: int = 5000):
    """Create 3D visualization of a specific structure type."""
    if not HAS_PLOTLY:
        print("⚠️  Plotly not available - skipping visualization")
        return
    
    print(f"\n📊 Creating visualization for {structure_type} structures...")
    
    # Filter to structure type
    filtered = filter_by_structure(df, structure_type, scale)
    if filtered is None or len(filtered) == 0:
        return
    
    # Sample if needed
    if len(filtered) > max_points:
        filtered = filtered.sample(n=max_points, shuffle=True)
        print(f"   (sampled {max_points:,} points for visualization)")
    
    # Find coordinate columns
    coord_cols = []
    for col in ['x', 'y', 'z']:
        if col in filtered.columns:
            coord_cols.append(col)
        else:
            for prefix in ['gaia', 'sdss', 'twomass']:
                prefixed_col = f"{prefix}_{col}"
                if prefixed_col in filtered.columns:
                    coord_cols.append(prefixed_col)
                    break
    
    if len(coord_cols) < 3:
        print(f"⚠️  Could not find 3D coordinates")
        return
    
    x = filtered[coord_cols[0]].to_numpy()
    y = filtered[coord_cols[1]].to_numpy()
    z = filtered[coord_cols[2]].to_numpy()
    
    # Create 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color='#ff7f0e',  # Orange for filaments
            opacity=0.7
        ),
        name=structure_type.capitalize(),
        hovertemplate=f'<b>{structure_type.capitalize()}</b><br>' +
                     'X: %{x:.1f} pc<br>' +
                     'Y: %{y:.1f} pc<br>' +
                     'Z: %{z:.1f} pc<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"{structure_type.capitalize()} Structures (Scale: {scale} pc)",
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
        font=dict(color="white"),
        height=800
    )
    
    # Save
    output_dir = Path("data/visualizations/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{structure_type}_structures.html"
    fig.write_html(output_path)
    print(f"   ✓ Saved to {output_path}")


def main():
    """Main example workflow."""
    print("\n" + "=" * 80)
    print("🌌 AstroLab Catalog Example")
    print("=" * 80)
    
    # Load catalog
    df = load_catalog()
    if df is None:
        return
    
    # Explore structure
    explore_catalog_structure(df)
    
    # Analyze cosmic web distribution
    analyze_cosmic_web_distribution(df, scale=10.0)
    
    # Compare scales
    compare_scales(df)
    
    # Filter examples
    print("\n" + "=" * 80)
    print("FILTERING EXAMPLES")
    print("=" * 80)
    
    # Example 1: Get filament sources
    filaments = filter_by_structure(df, "filament", scale=10.0)
    
    # Example 2: Get node (cluster) sources
    nodes = filter_by_structure(df, "node", scale=25.0)
    
    # Example 3: Get void sources
    voids = filter_by_structure(df, "void", scale=50.0)
    
    # Create visualizations
    if HAS_PLOTLY:
        print("\n" + "=" * 80)
        print("VISUALIZATIONS")
        print("=" * 80)
        
        visualize_structure_subset(df, "filament", scale=10.0)
        visualize_structure_subset(df, "node", scale=25.0)
    
    # Science case: High-density regions
    print("\n" + "=" * 80)
    print("SCIENCE CASE: High-Density Regions")
    print("=" * 80)
    
    density_col = "density_10.0pc"
    if density_col in df.columns:
        # Get high-density sources (top 10%)
        density_threshold = df[density_col].quantile(0.9)
        high_density = df.filter(pl.col(density_col) > density_threshold)
        
        print(f"\nHigh-density sources (top 10%): {len(high_density):,}")
        print(f"Density threshold: {density_threshold:.6f}")
        
        # What structure types are in high-density regions?
        class_col = "cosmic_web_class_10.0pc"
        if class_col in high_density.columns:
            print("\nStructure types in high-density regions:")
            counts = high_density[class_col].value_counts().sort(class_col)
            class_names = {0: "Field", 1: "Filament", 2: "Void", 3: "Node"}
            for row in counts.iter_rows():
                class_name = class_names.get(int(row[0]), f"Class {row[0]}")
                print(f"  {class_name}: {row[1]:,}")
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("  1. Try different filtering criteria")
    print("  2. Combine with photometric cuts")
    print("  3. Export subsets for detailed analysis")
    print("  4. Train ML models on structure classification")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
