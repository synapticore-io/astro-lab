"""
Example: Preprocessing and Combining Multiple Surveys
=====================================================

Shows how to:
1. Preprocess individual surveys (Gaia, SDSS, 2MASS)
2. Cross-match between surveys
3. Build a combined multi-wavelength graph
4. Generate consolidated AstroLab catalog with cosmic web features
5. Create visual outputs

This example demonstrates the complete workflow from raw data to 
publication-quality visualizations.
"""

import logging
from pathlib import Path

import polars as pl
import torch
from torch_geometric.data import Data

from astro_lab.data.cross_match import SurveyCrossMatcher
from astro_lab.data.preprocessors.gaia import GaiaPreprocessor
from astro_lab.data.preprocessors.sdss import SDSSPreprocessor
from astro_lab.data.preprocessors.twomass import TwoMASSPreprocessor
from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer

# Import visualization utilities
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available - visualizations will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate multi-survey preprocessing and cross-matching."""

    # 1. Preprocess individual surveys
    logger.info("=== Step 1: Preprocessing Individual Surveys ===")

    surveys = {}
    graphs = {}

    # Process Gaia (astrometry + optical)
    logger.info("\nProcessing Gaia DR3...")
    gaia_proc = GaiaPreprocessor()
    gaia_df, gaia_graph = gaia_proc.preprocess(max_samples=100000)
    surveys["gaia"] = gaia_df
    graphs["gaia"] = gaia_graph

    # Process SDSS (deep optical photometry)
    logger.info("\nProcessing SDSS...")
    sdss_proc = SDSSPreprocessor()
    sdss_df, sdss_graph = sdss_proc.preprocess(max_samples=50000)
    surveys["sdss"] = sdss_df
    graphs["sdss"] = sdss_graph

    # Process 2MASS (near-infrared)
    logger.info("\nProcessing 2MASS...")
    twomass_proc = TwoMASSPreprocessor()
    twomass_df, twomass_graph = twomass_proc.preprocess(max_samples=50000)
    surveys["twomass"] = twomass_df
    graphs["twomass"] = twomass_graph

    # 2. Cross-match surveys
    logger.info("\n=== Step 2: Cross-Matching Surveys ===")

    matcher = SurveyCrossMatcher(max_separation=1.0)  # 1 arcsec

    # Match all to Gaia as reference
    combined_df = matcher.multi_survey_match(surveys=surveys, reference_survey="gaia")

    logger.info(f"\nCombined catalog has {len(combined_df):,} sources")
    logger.info(f"Columns: {len(combined_df.columns)}")

    # 3. Create multi-wavelength features
    logger.info("\n=== Step 3: Creating Multi-Wavelength Features ===")

    # Build feature vector combining all surveys
    feature_cols = []

    # Gaia features
    gaia_features = [
        "gaia_phot_g_mean_mag",
        "gaia_phot_bp_mean_mag",
        "gaia_phot_rp_mean_mag",
        "gaia_bp_rp",
        "gaia_parallax",
        "gaia_pmra",
        "gaia_pmdec",
    ]
    feature_cols.extend([f for f in gaia_features if f in combined_df.columns])

    # SDSS features
    sdss_features = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    feature_cols.extend([f for f in sdss_features if f in combined_df.columns])

    # 2MASS features
    twomass_features = ["twomass_j_m", "twomass_h_m", "twomass_k_m"]
    feature_cols.extend([f for f in twomass_features if f in combined_df.columns])

    logger.info(f"Total features: {len(feature_cols)}")

    # 4. Build combined graph
    logger.info("\n=== Step 4: Building Combined Graph ===")

    # Use Gaia positions (most accurate)
    pos_cols = ["gaia_x", "gaia_y", "gaia_z"]

    # Filter to objects with positions and features
    mask = (
        combined_df.select(pos_cols + feature_cols).null_count().sum_horizontal() == 0
    )
    filtered_df = combined_df.filter(mask)

    logger.info(f"Filtered to {len(filtered_df):,} complete sources")

    # Extract data
    positions = torch.tensor(
        filtered_df.select(pos_cols).to_numpy(), dtype=torch.float32
    )

    features = torch.tensor(
        filtered_df.select(feature_cols).to_numpy(), dtype=torch.float32
    )

    # Build graph using Gaia preprocessor's method
    edge_index = gaia_proc._build_graph_structure(positions)

    # Create combined graph
    combined_graph = Data(x=features, pos=positions, edge_index=edge_index)

    # Add metadata
    combined_graph.feature_names = feature_cols
    combined_graph.surveys = ["gaia", "sdss", "twomass"]
    combined_graph.n_sources = len(filtered_df)

    # Add source IDs
    if "gaia_source_id" in filtered_df.columns:
        combined_graph.source_id = torch.tensor(
            filtered_df["gaia_source_id"].to_numpy(), dtype=torch.long
        )

    # 5. Save results
    logger.info("\n=== Step 5: Saving Results ===")

    output_dir = Path("data/processed/combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined catalog
    catalog_path = output_dir / "gaia_sdss_twomass.parquet"
    filtered_df.write_parquet(catalog_path)
    logger.info(f"Saved catalog to {catalog_path}")

    # Save combined graph
    graph_path = output_dir / "gaia_sdss_twomass.pt"
    torch.save(combined_graph, graph_path)
    logger.info(f"Saved graph to {graph_path}")

    # 6. Add cosmic web analysis
    logger.info("\n=== Step 6: Cosmic Web Analysis ===")
    
    analyzer = ScalableCosmicWebAnalyzer(max_points_per_batch=50000)
    clustering_scales = [5.0, 10.0, 25.0, 50.0]
    
    logger.info(f"Analyzing cosmic web at scales: {clustering_scales}")
    cw_results = analyzer.analyze_cosmic_web(
        coordinates=positions,
        scales=clustering_scales,
        use_adaptive_sampling=True
    )
    
    # Add cosmic web classifications to dataframe
    for i, scale in enumerate(clustering_scales):
        scale_key = f"scale_{scale:.1f}"
        if "multi_scale" in cw_results and scale_key in cw_results["multi_scale"]:
            scale_results = cw_results["multi_scale"][scale_key]
            
            if "structure_class" in scale_results:
                struct_class = scale_results["structure_class"].cpu().numpy()
                filtered_df = filtered_df.with_columns(
                    pl.Series(f"cosmic_web_class_{scale:.1f}pc", struct_class)
                )
            
            if "density" in scale_results:
                density = scale_results["density"].cpu().numpy()
                filtered_df = filtered_df.with_columns(
                    pl.Series(f"density_{scale:.1f}pc", density)
                )
    
    # Save enhanced catalog with cosmic web features
    enhanced_catalog_path = output_dir / "gaia_sdss_twomass_cosmicweb.parquet"
    filtered_df.write_parquet(enhanced_catalog_path)
    logger.info(f"Saved enhanced catalog to {enhanced_catalog_path}")
    
    # 7. Generate visualizations
    if HAS_PLOTLY:
        logger.info("\n=== Step 7: Generating Visualizations ===")
        
        vis_dir = Path("data/visualizations/examples")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 3D visualization (sample for performance)
        sample_size = min(5000, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, shuffle=True)
        
        x = sample_df.select(pos_cols[0]).to_numpy().flatten()
        y = sample_df.select(pos_cols[1]).to_numpy().flatten()
        z = sample_df.select(pos_cols[2]).to_numpy().flatten()
        
        # Get structure classification at 10 pc scale
        class_col = "cosmic_web_class_10.0pc"
        if class_col in sample_df.columns:
            classes = sample_df[class_col].to_numpy()
            class_names = {0: "Field", 1: "Filament", 2: "Void", 3: "Node"}
            colors_map = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
            
            labels = [class_names.get(int(c), f"Class {c}") for c in classes]
            colors = [colors_map.get(int(c), "#gray") for c in classes]
        else:
            labels = ["Source"] * len(sample_df)
            colors = ["#1f77b4"] * len(sample_df)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=colors, opacity=0.6),
            text=labels,
            hovertemplate='<b>%{text}</b><br>X: %{x:.1f} pc<br>Y: %{y:.1f} pc<br>Z: %{z:.1f} pc<extra></extra>'
        )])
        
        fig.update_layout(
            title="Multi-Survey Combined Catalog with Cosmic Web Structure",
            scene=dict(
                xaxis_title="X (parsec)",
                yaxis_title="Y (parsec)",
                zaxis_title="Z (parsec)",
                bgcolor="black",
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            height=800
        )
        
        vis_path = vis_dir / "combined_catalog_3d.html"
        fig.write_html(vis_path)
        logger.info(f"Saved 3D visualization to {vis_path}")
    
    # 8. Print statistics
    logger.info("\n=== Final Statistics ===")
    logger.info(f"Nodes: {combined_graph.num_nodes}")
    logger.info(f"Edges: {combined_graph.num_edges}")
    logger.info(f"Features: {combined_graph.x.shape[1]}")
    logger.info(
        f"Average degree: {combined_graph.num_edges / combined_graph.num_nodes:.1f}"
    )

    # Color statistics (optical to NIR)
    if all(f in feature_cols for f in ["gaia_phot_g_mean_mag", "twomass_k_m"]):
        g_idx = feature_cols.index("gaia_phot_g_mean_mag")
        k_idx = feature_cols.index("twomass_k_m")
        g_k_color = features[:, g_idx] - features[:, k_idx]
        logger.info(f"G-K color range: [{g_k_color.min():.2f}, {g_k_color.max():.2f}]")

    return combined_graph


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Multi-Survey Preprocessing and Cosmic Web Analysis")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Preprocessing multiple astronomical surveys")
    print("  2. Cross-matching between surveys")
    print("  3. Building combined multi-wavelength graphs")
    print("  4. Cosmic web structure analysis")
    print("  5. Visual output generation")
    print("\n" + "=" * 80 + "\n")
    
    combined_graph = main()

    # Example: Select red giants using multi-wavelength data
    print("\n=== Example: Finding Red Giants ===")

    # Red giants typically have:
    # - Red optical colors (BP-RP > 1.0)
    # - Bright in infrared (K < 10)
    # - Low parallax (distant)

    if hasattr(combined_graph, "feature_names"):
        feature_names = combined_graph.feature_names

        # Find relevant indices
        bp_rp_idx = (
            feature_names.index("gaia_bp_rp") if "gaia_bp_rp" in feature_names else None
        )
        k_idx = (
            feature_names.index("twomass_k_m")
            if "twomass_k_m" in feature_names
            else None
        )
        parallax_idx = (
            feature_names.index("gaia_parallax")
            if "gaia_parallax" in feature_names
            else None
        )

        if all(idx is not None for idx in [bp_rp_idx, k_idx, parallax_idx]):
            # Select red giants
            red_giants_mask = (
                (combined_graph.x[:, bp_rp_idx] > 1.0)  # Red color
                & (combined_graph.x[:, k_idx] < 10.0)  # Bright in K
                & (combined_graph.x[:, parallax_idx] < 1.0)  # Distant (> 1 kpc)
            )

            n_red_giants = red_giants_mask.sum().item()
            print(f"Found {n_red_giants:,} potential red giants")
            print(f"({n_red_giants / combined_graph.num_nodes * 100:.1f}% of sample)")
    
    print("\n" + "=" * 80)
    print("💡 Next Steps:")
    print("=" * 80)
    print("\n1. Generate full AstroLab catalog:")
    print("   python scripts/generate_astrolab_catalog.py --max-samples 100000")
    print("\n2. Create visualizations:")
    print("   python scripts/generate_visualizations.py")
    print("\n3. Explore the catalog:")
    print("   import polars as pl")
    print("   catalog = pl.read_parquet('data/catalogs/astrolab_catalog_v1.parquet')")
    print("\n" + "=" * 80 + "\n")
