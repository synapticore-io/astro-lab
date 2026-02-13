#!/usr/bin/env python3
"""
Generate Sample Astronomical Data
==================================

Creates synthetic astronomical data for demonstration purposes.
This is used when real data download is not available.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_sample_gaia_data(n_sources: int = 10000) -> pl.DataFrame:
    """
    Generate synthetic Gaia-like data.
    
    Args:
        n_sources: Number of sources to generate
        
    Returns:
        Polars DataFrame with Gaia-like columns
    """
    logger.info(f"🎲 Generating {n_sources:,} synthetic Gaia sources...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate RA, Dec covering a region
    ra = np.random.uniform(0, 360, n_sources)
    dec = np.random.uniform(-90, 90, n_sources)
    
    # Generate distances (parallax -> distance in parsecs)
    parallax_mas = np.random.uniform(0.5, 50, n_sources)  # 0.5 to 50 mas (20 to 2000 pc)
    distance_pc = 1000.0 / parallax_mas  # Convert to parsecs
    
    # Generate magnitudes
    g_mag = np.random.uniform(8, 12, n_sources)
    bp_mag = g_mag + np.random.uniform(-0.5, 1.0, n_sources)
    rp_mag = g_mag + np.random.uniform(-0.5, 1.0, n_sources)
    
    # Generate proper motions
    pmra = np.random.normal(0, 10, n_sources)  # mas/yr
    pmdec = np.random.normal(0, 10, n_sources)  # mas/yr
    
    # Convert to 3D Cartesian coordinates
    # Simplified conversion (not using exact spherical coordinates)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance_pc * np.sin(dec_rad)
    
    # Create DataFrame
    df = pl.DataFrame({
        'source_id': np.arange(n_sources, dtype=np.int64),
        'ra': ra,
        'dec': dec,
        'parallax': parallax_mas,
        'distance_pc': distance_pc,
        'phot_g_mean_mag': g_mag,
        'phot_bp_mean_mag': bp_mag,
        'phot_rp_mean_mag': rp_mag,
        'pmra': pmra,
        'pmdec': pmdec,
        'x': x,
        'y': y,
        'z': z,
    })
    
    logger.info(f"   ✓ Generated {len(df):,} sources")
    logger.info(f"   ✓ Distance range: {distance_pc.min():.1f} - {distance_pc.max():.1f} pc")
    logger.info(f"   ✓ Position range X: {x.min():.1f} - {x.max():.1f} pc")
    
    return df


def save_sample_data(df: pl.DataFrame, output_dir: Path = Path("data/raw")):
    """Save sample data to parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "gaia_dr3_bright_all_sky_mag12.0.parquet"
    df.write_parquet(output_path, compression="zstd")
    
    logger.info(f"\n💾 Saved sample data:")
    logger.info(f"   {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Also save metadata
    metadata = {
        "survey": "gaia_dr3_sample",
        "n_sources": len(df),
        "magnitude_limit": 12.0,
        "note": "Synthetic data for demonstration"
    }
    
    import json
    metadata_path = output_dir / "gaia_dr3_bright_all_sky_mag12.0.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   {metadata_path}")
    
    return output_path


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("🌌 Sample Data Generation")
    logger.info("="*80)
    
    # Generate sample data
    df = generate_sample_gaia_data(n_sources=10000)
    
    # Save to file
    output_path = save_sample_data(df)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Sample data generation complete!")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
