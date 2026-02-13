# AstroLab Data Pipeline Guide

This guide explains how to use the AstroLab data acquisition and catalog generation pipeline.

## Quick Start

Generate sample data, create a consolidated catalog, and visualize it:

```bash
# Step 1: Generate sample astronomical data (if you don't have real data)
python scripts/generate_sample_data.py

# Step 2: Run the complete pipeline
PYTHONPATH=src python scripts/complete_data_pipeline.py --max-samples 5000 --skip-download

# Or download real data and process it (requires network access)
PYTHONPATH=src python scripts/complete_data_pipeline.py --max-samples 5000
```

## Generated Files

### Raw Data (`data/raw/`)
- `gaia_dr3_bright_all_sky_mag12.0.parquet` - Raw Gaia DR3 data (10,000 sources)
- `gaia_dr3_bright_all_sky_mag12.0.json` - Metadata about the raw data

### Catalogs (`data/catalogs/`)
- `astrolab_catalog_v1.parquet` - Full consolidated catalog with processed features
- `astrolab_catalog_v1_sample.parquet` - Sample catalog for quick testing

### Visualizations (`data/visualizations/`)
- `cosmic_web_3d.html` - Interactive 3D scatter plot of astronomical sources

## Catalog Schema

The AstroLab catalog includes the following features:

### Astrometric Features
- `source_id` - Unique identifier
- `ra`, `dec` - Right ascension and declination (degrees)
- `gal_l`, `gal_b` - Galactic coordinates (degrees)
- `parallax` - Parallax (milliarcseconds)
- `distance_pc` - Distance in parsecs
- `x`, `y`, `z` - 3D Cartesian coordinates (parsecs)

### Photometric Features
- `phot_g_mean_mag` - G-band magnitude
- `phot_bp_mean_mag` - BP-band magnitude
- `phot_rp_mean_mag` - RP-band magnitude
- `bp_rp` - BP-RP color index
- `g_bp` - G-BP color index
- `g_rp` - G-RP color index
- `mg_abs` - Absolute G magnitude

### Kinematic Features
- `pmra`, `pmdec` - Proper motions (mas/yr)
- `pm_total` - Total proper motion (mas/yr)
- `v_tan_kms` - Tangential velocity (km/s)

### Derived Features
- `stellar_mass_est` - Estimated stellar mass

### Metadata
- `catalog_version` - Catalog version (v1.0)
- `processing_date` - ISO timestamp of processing

## Using the Catalog

### Load with Polars

```python
import polars as pl

# Load full catalog
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")
print(f"Loaded {len(catalog):,} sources")

# Filter by magnitude
bright_sources = catalog.filter(pl.col("phot_g_mean_mag") < 10)

# Select nearby sources (within 100 parsecs)
nearby = catalog.filter(pl.col("distance_pc") < 100)
```

### Load with Pandas

```python
import pandas as pd

# Load full catalog
catalog = pd.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")
print(f"Loaded {len(catalog):,} sources")
```

### 3D Visualization

Open `data/visualizations/cosmic_web_3d.html` in a web browser to explore the 3D distribution of astronomical sources.

## Pipeline Components

### 1. Sample Data Generation (`scripts/generate_sample_data.py`)

Creates synthetic Gaia-like astronomical data for demonstration:
- Generates realistic RA, Dec, parallax, proper motions
- Converts to 3D Cartesian coordinates
- Saves as Parquet for efficient processing

### 2. Complete Pipeline (`scripts/complete_data_pipeline.py`)

Orchestrates the full data pipeline:
1. Downloads real survey data (or uses sample data)
2. Preprocesses and cleans the data
3. Runs cosmic web structure analysis
4. Generates consolidated catalog
5. Creates visualizations

Options:
- `--max-samples N` - Process only N samples
- `--skip-download` - Use existing data files
- `--force-download` - Force re-download of data
- `--surveys gaia sdss` - Specify which surveys to include

### 3. Catalog Generation (`scripts/generate_astrolab_catalog.py`)

Standalone script for generating catalogs from multiple surveys:
- Cross-matches multiple surveys
- Runs cosmic web analysis at multiple scales
- Adds structure classifications (filaments, voids, nodes)
- Computes density fields and anisotropy metrics

### 4. Visualization Generation (`scripts/generate_visualizations.py`)

Creates publication-quality visualizations:
- Interactive 3D scatter plots
- Structure distribution plots
- Multi-scale comparison plots
- Density distribution histograms

## Known Issues

### Cosmic Web Analysis

The cosmic web analysis may fail with tensor indexing errors when processing certain data formats. The pipeline gracefully handles this by:
1. Warning about the failure
2. Continuing without cosmic web features
3. Still generating the catalog with astrometric and photometric features

To fix this, the tensor format expected by `ScalableCosmicWebAnalyzer` needs to be adjusted.

### Network Requirements

Downloading real data requires:
- Network access to astronomical archives
- Sufficient bandwidth for large catalogs
- Proper firewall configuration

If network access is unavailable, use the sample data generator instead.

## Extending the Pipeline

### Add New Surveys

1. Create a collector in `src/astro_lab/data/collectors/your_survey.py`
2. Implement the `download()` method
3. Create a preprocessor in `src/astro_lab/data/preprocessors/your_survey.py`
4. Add to the pipeline in `complete_data_pipeline.py`

### Add New Features

1. Modify the preprocessor to compute new features
2. Update the catalog schema documentation
3. Add visualization support if needed

### Customize Analysis

The cosmic web analysis parameters can be adjusted:
- `clustering_scales` - Scales for structure detection (parsecs)
- `max_points_per_batch` - Batch size for large datasets
- `use_adaptive_sampling` - Enable adaptive sampling for speed

## Performance Tips

1. **Sampling**: Use `--max-samples` for quick iteration
2. **Caching**: Downloaded data is cached in `data/raw/`
3. **Parallel Processing**: The pipeline automatically uses available CPU cores
4. **Memory**: Large catalogs (>1M sources) may require 16GB+ RAM

## Data Quality

The preprocessor applies quality filters:
- Parallax signal-to-noise ratio > 5 (if available)
- Astrometric excess noise < 1.0 (if available)
- RUWE (Renormalized Unit Weight Error) < 1.4 (if available)

## Citations

If you use the AstroLab catalog in research, please cite:

```bibtex
@software{astrolab_catalog,
  title = {AstroLab Catalog: Multi-Survey Cosmic Web Dataset},
  author = {AstroLab Team},
  year = {2026},
  url = {https://github.com/synapticore-io/astro-lab},
  version = {1.0}
}
```

Also cite the original surveys:
- **Gaia**: https://www.cosmos.esa.int/gaia
- **SDSS**: https://www.sdss.org/
- **2MASS**: https://www.ipac.caltech.edu/2mass/

## Support

For issues or questions:
1. Check this guide first
2. Review the example scripts
3. Open an issue on GitHub
