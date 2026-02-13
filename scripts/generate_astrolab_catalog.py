#!/usr/bin/env python3
"""
Generate Consolidated AstroLab Catalog
=======================================

Creates a consolidated parquet catalog combining multiple astronomical surveys
with cosmic web structure classifications.

Features:
- Multi-survey cross-matching (Gaia, SDSS, 2MASS)
- Cosmic web classification (filaments, voids, nodes)
- 3D coordinate conversion
- Density field computation
- Multi-scale structure analysis

Usage:
    python scripts/generate_astrolab_catalog.py [--max-samples N] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch

from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer
from astro_lab.data.preprocessors.gaia import GaiaPreprocessor
from astro_lab.data.preprocessors.sdss import SDSSPreprocessor
from astro_lab.data.preprocessors.twomass import TwoMASSPreprocessor
from astro_lab.data.cross_match import SurveyCrossMatcher
from astro_lab.tensors import SpatialTensorDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_catalog(
    max_samples: Optional[int] = None,
    output_dir: Path = Path("data/catalogs"),
    clustering_scales: List[float] = None,
    include_surveys: List[str] = None,
) -> Path:
    """
    Generate consolidated AstroLab catalog with cosmic web features.
    
    Args:
        max_samples: Maximum number of samples to process (None = all)
        output_dir: Directory to save the catalog
        clustering_scales: Scales for cosmic web clustering in parsecs
        include_surveys: List of surveys to include (default: all available)
        
    Returns:
        Path to the generated catalog file
    """
    if clustering_scales is None:
        clustering_scales = [5.0, 10.0, 25.0, 50.0]
    
    if include_surveys is None:
        include_surveys = ["gaia", "sdss", "twomass"]
    
    logger.info("=" * 80)
    logger.info("🌌 AstroLab Catalog Generation")
    logger.info("=" * 80)
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Clustering scales: {clustering_scales}")
    logger.info(f"Surveys: {', '.join(include_surveys)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocess individual surveys
    logger.info("\n📊 Step 1: Preprocessing Individual Surveys")
    logger.info("-" * 80)
    
    surveys = {}
    graphs = {}
    
    if "gaia" in include_surveys:
        logger.info("\n🌟 Processing Gaia DR3...")
        try:
            gaia_proc = GaiaPreprocessor()
            gaia_df, gaia_graph = gaia_proc.preprocess(max_samples=max_samples)
            surveys["gaia"] = gaia_df
            graphs["gaia"] = gaia_graph
            logger.info(f"   ✓ Gaia: {len(gaia_df):,} sources")
        except Exception as e:
            logger.warning(f"   ⚠ Gaia preprocessing failed: {e}")
    
    if "sdss" in include_surveys:
        logger.info("\n🔭 Processing SDSS...")
        try:
            sdss_proc = SDSSPreprocessor()
            sdss_df, sdss_graph = sdss_proc.preprocess(max_samples=max_samples)
            surveys["sdss"] = sdss_df
            graphs["sdss"] = sdss_graph
            logger.info(f"   ✓ SDSS: {len(sdss_df):,} sources")
        except Exception as e:
            logger.warning(f"   ⚠ SDSS preprocessing failed: {e}")
    
    if "twomass" in include_surveys:
        logger.info("\n📡 Processing 2MASS...")
        try:
            twomass_proc = TwoMASSPreprocessor()
            twomass_df, twomass_graph = twomass_proc.preprocess(max_samples=max_samples)
            surveys["twomass"] = twomass_df
            graphs["twomass"] = twomass_graph
            logger.info(f"   ✓ 2MASS: {len(twomass_df):,} sources")
        except Exception as e:
            logger.warning(f"   ⚠ 2MASS preprocessing failed: {e}")
    
    if not surveys:
        raise RuntimeError("No surveys were successfully preprocessed")
    
    # Step 2: Cross-match surveys (if multiple surveys)
    if len(surveys) > 1:
        logger.info("\n🔗 Step 2: Cross-Matching Surveys")
        logger.info("-" * 80)
        
        matcher = SurveyCrossMatcher(max_separation=1.0)  # 1 arcsec
        reference_survey = "gaia" if "gaia" in surveys else list(surveys.keys())[0]
        
        logger.info(f"Reference survey: {reference_survey}")
        combined_df = matcher.multi_survey_match(
            surveys=surveys,
            reference_survey=reference_survey
        )
        logger.info(f"   ✓ Combined catalog: {len(combined_df):,} matched sources")
    else:
        # Single survey - just use it directly
        survey_name = list(surveys.keys())[0]
        combined_df = surveys[survey_name]
        logger.info(f"\n📋 Using single survey: {survey_name}")
    
    # Step 3: Cosmic web analysis
    logger.info("\n🕸️  Step 3: Cosmic Web Structure Analysis")
    logger.info("-" * 80)
    
    # Get 3D coordinates from the combined dataframe
    coord_cols = []
    for col in ['x', 'y', 'z']:
        # Check for survey-prefixed columns first
        for survey in surveys.keys():
            if f"{survey}_{col}" in combined_df.columns:
                coord_cols.append(f"{survey}_{col}")
                break
        else:
            # Fall back to unprefixed columns
            if col in combined_df.columns:
                coord_cols.append(col)
    
    if len(coord_cols) != 3:
        raise ValueError(f"Could not find 3D coordinates in combined dataframe. Available columns: {combined_df.columns}")
    
    logger.info(f"Using coordinates: {coord_cols}")
    
    # Extract coordinates as tensor
    coords_array = combined_df.select(coord_cols).to_numpy()
    coordinates = torch.tensor(coords_array, dtype=torch.float32)
    
    # Run cosmic web analysis
    analyzer = ScalableCosmicWebAnalyzer(max_points_per_batch=100000)
    
    logger.info(f"Analyzing {coordinates.shape[0]:,} sources at {len(clustering_scales)} scales")
    cw_results = analyzer.analyze_cosmic_web(
        coordinates=coordinates,
        scales=clustering_scales,
        use_adaptive_sampling=True
    )
    
    logger.info("   ✓ Cosmic web analysis complete")
    
    # Step 4: Add cosmic web features to catalog
    logger.info("\n📝 Step 4: Adding Cosmic Web Features to Catalog")
    logger.info("-" * 80)
    
    # Get combined cosmic web classifications
    # For multi-scale analysis, we'll use the results from each scale
    for i, scale in enumerate(clustering_scales):
        scale_key = f"scale_{scale:.1f}"
        if "multi_scale" in cw_results and scale_key in cw_results["multi_scale"]:
            scale_results = cw_results["multi_scale"][scale_key]
            
            # Add structure classifications
            if "structure_class" in scale_results:
                struct_class = scale_results["structure_class"].cpu().numpy()
                combined_df = combined_df.with_columns(
                    pl.Series(f"cosmic_web_class_{scale:.1f}pc", struct_class)
                )
            
            # Add density field
            if "density" in scale_results:
                density = scale_results["density"].cpu().numpy()
                combined_df = combined_df.with_columns(
                    pl.Series(f"density_{scale:.1f}pc", density)
                )
            
            # Add anisotropy
            if "anisotropy" in scale_results:
                anisotropy = scale_results["anisotropy"].cpu().numpy()
                combined_df = combined_df.with_columns(
                    pl.Series(f"anisotropy_{scale:.1f}pc", anisotropy)
                )
    
    logger.info(f"   ✓ Added cosmic web features at {len(clustering_scales)} scales")
    
    # Step 5: Add metadata columns
    logger.info("\n📊 Step 5: Adding Metadata")
    logger.info("-" * 80)
    
    # Add catalog version
    combined_df = combined_df.with_columns(
        pl.lit("v1.0").alias("catalog_version")
    )
    
    # Add processing timestamp
    from datetime import datetime
    combined_df = combined_df.with_columns(
        pl.lit(datetime.now().isoformat()).alias("processing_date")
    )
    
    logger.info("   ✓ Metadata added")
    
    # Step 6: Save catalog
    logger.info("\n💾 Step 6: Saving Catalog")
    logger.info("-" * 80)
    
    catalog_path = output_dir / "astrolab_catalog_v1.parquet"
    combined_df.write_parquet(catalog_path, compression="zstd")
    
    # Also save a smaller sample for quick testing
    sample_size = min(10000, len(combined_df))
    sample_path = output_dir / "astrolab_catalog_v1_sample.parquet"
    combined_df.head(sample_size).write_parquet(sample_path, compression="zstd")
    
    logger.info(f"   ✓ Full catalog: {catalog_path}")
    logger.info(f"   ✓ Sample catalog: {sample_path}")
    
    # Step 7: Generate catalog statistics
    logger.info("\n📈 Step 7: Catalog Statistics")
    logger.info("=" * 80)
    
    logger.info(f"Total sources: {len(combined_df):,}")
    logger.info(f"Total columns: {len(combined_df.columns)}")
    logger.info(f"File size: {catalog_path.stat().st_size / (1024**2):.1f} MB")
    
    # Survey breakdown
    logger.info("\nSurveys included:")
    for survey in surveys.keys():
        logger.info(f"  - {survey.upper()}")
    
    # Cosmic web statistics
    logger.info("\nCosmic web features:")
    for scale in clustering_scales:
        col_name = f"cosmic_web_class_{scale:.1f}pc"
        if col_name in combined_df.columns:
            counts = combined_df[col_name].value_counts()
            logger.info(f"  Scale {scale:.1f} pc:")
            for row in counts.iter_rows():
                logger.info(f"    - Class {row[0]}: {row[1]:,} sources")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Catalog generation complete!")
    logger.info("=" * 80)
    
    return catalog_path


def main():
    """CLI entry point for catalog generation."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated AstroLab catalog with cosmic web features"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/catalogs"),
        help="Output directory for catalog (default: data/catalogs)"
    )
    parser.add_argument(
        "--clustering-scales",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 25.0, 50.0],
        help="Clustering scales in parsecs (default: 5 10 25 50)"
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        default=["gaia"],
        help="Surveys to include (default: gaia)"
    )
    
    args = parser.parse_args()
    
    try:
        catalog_path = generate_catalog(
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            clustering_scales=args.clustering_scales,
            include_surveys=args.surveys
        )
        
        print(f"\n✅ Success! Catalog saved to: {catalog_path}")
        print(f"\n📖 Load the catalog with:")
        print(f"    import polars as pl")
        print(f"    catalog = pl.read_parquet('{catalog_path}')")
        
    except Exception as e:
        logger.error(f"❌ Catalog generation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
