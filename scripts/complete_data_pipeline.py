#!/usr/bin/env python3
"""
Complete Data Acquisition and Catalog Generation Pipeline
==========================================================

This script performs the complete data pipeline:
1. Downloads astronomical survey data (Gaia)
2. Generates consolidated AstroLab catalog with cosmic web features
3. Creates visualizations from the catalog
4. Stores everything in the data/ directory structure

Usage:
    python scripts/complete_data_pipeline.py [--max-samples N] [--skip-download]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_lab.data.collectors.gaia import GaiaCollector
from astro_lab.data.preprocessors.gaia import GaiaPreprocessor
from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer

# Import visualization functions
sys.path.insert(0, str(Path(__file__).parent))
from generate_visualizations import generate_all_visualizations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_survey_data(survey: str = "gaia", force: bool = False) -> list:
    """
    Download survey data.
    
    Args:
        survey: Survey name (default: gaia)
        force: Force re-download
        
    Returns:
        List of downloaded file paths
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"📥 Step 1: Downloading {survey.upper()} Data")
    logger.info(f"{'='*80}")
    
    if survey.lower() == "gaia":
        collector = GaiaCollector(survey)
        # Use a reasonable magnitude limit for quick download
        collector.magnitude_limit = 12.0
        downloaded_files = collector.download(force=force)
        
        logger.info(f"\n✅ Downloaded {len(downloaded_files)} file(s)")
        for file_path in downloaded_files:
            logger.info(f"   📄 {file_path}")
        
        return downloaded_files
    else:
        raise ValueError(f"Survey '{survey}' not yet supported in this pipeline")


def generate_simple_catalog(
    max_samples: Optional[int] = None,
    output_dir: Path = Path("data/catalogs"),
    clustering_scales: list = None,
    batch_size: int = 100000,
    sample_catalog_size: int = 10000
) -> Path:
    """
    Generate consolidated AstroLab catalog with cosmic web features.
    
    Args:
        max_samples: Maximum number of samples to process
        output_dir: Output directory for catalog
        clustering_scales: Scales for cosmic web clustering in parsecs
        batch_size: Batch size for cosmic web analysis (adjust based on memory)
        sample_catalog_size: Size of sample catalog to generate
        
    Returns:
        Path to generated catalog
    """
    if clustering_scales is None:
        clustering_scales = [5.0, 10.0, 25.0]
    
    logger.info("\n📊 Processing Gaia Data...")
    
    # Load raw Gaia data
    raw_dir = Path("data/raw")
    gaia_file = raw_dir / "gaia_dr3_bright_all_sky_mag12.0.parquet"
    
    if not gaia_file.exists():
        raise FileNotFoundError(f"Gaia data not found: {gaia_file}")
    
    logger.info(f"   Loading from {gaia_file}")
    gaia_df = pl.read_parquet(gaia_file)
    
    # Sample if requested
    if max_samples and len(gaia_df) > max_samples:
        logger.info(f"   Sampling {max_samples:,} of {len(gaia_df):,} sources")
        gaia_df = gaia_df.sample(n=max_samples, shuffle=True, seed=42)
    
    # Preprocess
    gaia_proc = GaiaPreprocessor()
    gaia_df = gaia_proc.preprocess(gaia_df)
    logger.info(f"   ✓ Processed {len(gaia_df):,} sources")
    
    # Ensure we have 3D coordinates
    required_cols = ['x', 'y', 'z']
    if not all(col in gaia_df.columns for col in required_cols):
        # Coordinates should already be in the data, but double-check
        logger.info("   Adding 3D coordinates...")
        ra = gaia_df['ra'].to_numpy()
        dec = gaia_df['dec'].to_numpy()
        distance_pc = gaia_df['distance_pc'].to_numpy()
        
        import numpy as np
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance_pc * np.sin(dec_rad)
        
        gaia_df = gaia_df.with_columns([
            pl.Series('x', x),
            pl.Series('y', y),
            pl.Series('z', z)
        ])
    
    # Get 3D coordinates
    logger.info("\n🕸️  Running Cosmic Web Analysis...")
    coord_cols = ['x', 'y', 'z']
    coords_array = gaia_df.select(coord_cols).to_numpy()
    coordinates = torch.tensor(coords_array, dtype=torch.float32)
    
    logger.info(f"   Coordinates shape: {coordinates.shape}")
    
    # Run cosmic web analysis
    try:
        analyzer = ScalableCosmicWebAnalyzer(max_points_per_batch=batch_size)
        cw_results = analyzer.analyze_cosmic_web(
            coordinates=coordinates,
            scales=clustering_scales,
            use_adaptive_sampling=True
        )
        logger.info("   ✓ Cosmic web analysis complete")
    except Exception as e:
        logger.warning(f"   ⚠ Cosmic web analysis failed: {e}")
        logger.info("   Continuing without cosmic web features...")
        cw_results = None
    
    # Add cosmic web features to catalog
    logger.info("\n📝 Adding Cosmic Web Features...")
    if cw_results:
        for i, scale in enumerate(clustering_scales):
            scale_key = f"scale_{scale:.1f}"
            if "multi_scale" in cw_results and scale_key in cw_results["multi_scale"]:
                scale_results = cw_results["multi_scale"][scale_key]
                
                # Add structure classifications
                if "structure_class" in scale_results:
                    struct_class = scale_results["structure_class"].cpu().numpy()
                    gaia_df = gaia_df.with_columns(
                        pl.Series(f"cosmic_web_class_{scale:.1f}pc", struct_class)
                    )
                
                # Add density field
                if "density" in scale_results:
                    density = scale_results["density"].cpu().numpy()
                    gaia_df = gaia_df.with_columns(
                        pl.Series(f"density_{scale:.1f}pc", density)
                    )
                
                # Add anisotropy
                if "anisotropy" in scale_results:
                    anisotropy = scale_results["anisotropy"].cpu().numpy()
                    gaia_df = gaia_df.with_columns(
                        pl.Series(f"anisotropy_{scale:.1f}pc", anisotropy)
                    )
        
        logger.info(f"   ✓ Added features at {len(clustering_scales)} scales")
    else:
        logger.info("   ⚠ Skipped (cosmic web analysis not available)")
    
    # Add metadata
    logger.info("\n📊 Adding Metadata...")
    gaia_df = gaia_df.with_columns([
        pl.lit("v1.0").alias("catalog_version"),
        pl.lit(datetime.now().isoformat()).alias("processing_date")
    ])
    
    # Save catalog
    logger.info("\n💾 Saving Catalog...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    catalog_path = output_dir / "astrolab_catalog_v1.parquet"
    gaia_df.write_parquet(catalog_path, compression="zstd")
    
    # Save sample
    sample_size = min(sample_catalog_size, len(gaia_df))
    sample_path = output_dir / "astrolab_catalog_v1_sample.parquet"
    gaia_df.head(sample_size).write_parquet(sample_path, compression="zstd")
    
    logger.info(f"   ✓ Full catalog: {catalog_path}")
    logger.info(f"   ✓ Sample catalog: {sample_path}")
    logger.info(f"   ✓ Total sources: {len(gaia_df):,}")
    logger.info(f"   ✓ File size: {catalog_path.stat().st_size / (1024**2):.1f} MB")
    
    return catalog_path


def main():
    """Main entry point for complete data pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete data acquisition and catalog generation pipeline"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to process (default: 10000)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download step (use existing data)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of data"
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        default=["gaia"],
        help="Surveys to include (default: gaia)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("\n" + "="*80)
        logger.info("🌌 AstroLab Complete Data Pipeline")
        logger.info("="*80)
        logger.info(f"Configuration:")
        logger.info(f"  Max samples: {args.max_samples:,}")
        logger.info(f"  Surveys: {', '.join(args.surveys)}")
        logger.info(f"  Skip download: {args.skip_download}")
        
        # Step 1: Download data (if not skipped)
        if not args.skip_download:
            for survey in args.surveys:
                download_survey_data(survey, force=args.force_download)
        else:
            logger.info("\n⏩ Skipping download step (using existing data)")
        
        # Step 2: Generate catalog
        logger.info(f"\n{'='*80}")
        logger.info("📊 Step 2: Generating AstroLab Catalog")
        logger.info(f"{'='*80}")
        
        catalog_path = generate_simple_catalog(
            max_samples=args.max_samples,
            output_dir=Path("data/catalogs"),
            clustering_scales=[5.0, 10.0, 25.0]
        )
        
        logger.info(f"\n✅ Catalog generated: {catalog_path}")
        
        # Step 3: Generate visualizations
        logger.info(f"\n{'='*80}")
        logger.info("🎨 Step 3: Generating Visualizations")
        logger.info(f"{'='*80}")
        
        generate_all_visualizations(
            catalog_path=catalog_path,
            output_dir=Path("data/visualizations")
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info("\n📁 Output files:")
        
        # List catalogs
        logger.info("\n📊 Catalogs:")
        for file in sorted(Path("data/catalogs").glob("*.parquet")):
            size_mb = file.stat().st_size / (1024**2)
            logger.info(f"   {file.name} ({size_mb:.2f} MB)")
        
        # List visualizations
        logger.info("\n🎨 Visualizations:")
        for file in sorted(Path("data/visualizations").glob("*.html")):
            size_kb = file.stat().st_size / 1024
            logger.info(f"   {file.name} ({size_kb:.1f} KB)")
        
        logger.info("\n📖 Next steps:")
        logger.info("   1. Open HTML files in data/visualizations/ to view 3D cosmic web")
        logger.info("   2. Load catalog with: polars.read_parquet('data/catalogs/astrolab_catalog_v1.parquet')")
        logger.info("   3. Explore cosmic web classifications and density fields")
        
        return 0
        
    except KeyboardInterrupt:
        logger.error("\n❌ Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
