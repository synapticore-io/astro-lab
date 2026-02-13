# AstroLab Data Directory

This directory contains astronomical data, processed catalogs, and visualizations for the AstroLab project.

## Directory Structure

```
data/
├── raw/              # Raw astronomical survey data (downloaded from archives)
├── processed/        # Preprocessed and harmonized survey data
├── catalogs/         # Consolidated AstroLab catalogs ready for analysis
└── visualizations/   # Generated visualizations and plots
```

## Directory Descriptions

### `raw/`
Contains raw astronomical survey data downloaded from various archives:
- **Gaia DR3**: Astrometric and photometric data for stars
- **SDSS**: Optical spectroscopy and photometry for galaxies
- **2MASS**: Near-infrared photometry
- **NASA archives**: Various astronomical catalogs

Files in this directory are typically in their original formats (FITS, CSV, Parquet).

### `processed/`
Contains preprocessed data after harmonization and cleaning:
- Standardized column names (ra, dec, distance_pc, x, y, z)
- Quality filtering applied
- 3D coordinates computed
- Missing values handled
- Saved as Parquet files for efficient processing

### `catalogs/`
Contains consolidated AstroLab catalogs combining multiple surveys with cosmic web features:

#### AstroLab Catalog v1.0
**File**: `astrolab_catalog_v1.parquet`

A comprehensive multi-survey catalog with cosmic web structure classifications.

**Surveys included**:
- Gaia DR3 (astrometry, optical photometry)
- SDSS (deep optical photometry and spectroscopy)
- 2MASS (near-infrared photometry)

**Features**:
- **Astrometry**: Position (RA, Dec, distance), proper motion, parallax
- **Photometry**: Multi-wavelength magnitudes (optical + infrared)
- **3D Coordinates**: Cartesian coordinates (x, y, z) in parsecs
- **Cosmic Web Classification**: Filament, void, node, field classifications at multiple scales
- **Density Field**: Local density estimates
- **Structure Properties**: Anisotropy, connectivity, topology metrics

**Column Schema**: See `catalog_schema.md` for detailed column descriptions.

**Usage**:
```python
import polars as pl
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")
```

### `visualizations/`
Contains generated visualizations from cosmic web analysis:
- **3D Scatter Plots**: Interactive HTML plots (Plotly)
- **Structure Maps**: PNG/PDF plots of cosmic web structures
- **Multi-Scale Analysis**: Comparative visualizations at different scales
- **Statistical Plots**: Distribution and correlation plots

## Data Size and Git LFS

Large data files (> 100 MB) are excluded from git tracking via `.gitignore`.

If you need to share large datasets:
1. Use external storage (e.g., Zenodo, Figshare)
2. Use Git LFS for datasets < 1 GB
3. Document download instructions in this README

## Generating the AstroLab Catalog

To generate the consolidated AstroLab catalog:

```bash
# Generate catalog from cosmic web analysis
python scripts/generate_astrolab_catalog.py

# Or use the CLI
astro-lab generate-catalog --surveys gaia sdss twomass --output data/catalogs/
```

## Citing the AstroLab Catalog

If you use the AstroLab catalog in your research, please cite:

```bibtex
@software{astrolab_catalog,
  title = {AstroLab Catalog: Multi-Survey Cosmic Web Dataset},
  author = {AstroLab Team},
  year = {2026},
  url = {https://github.com/synapticore-io/astro-lab},
  version = {1.0}
}
```

## Data Sources and Acknowledgments

- **Gaia**: ESA Gaia mission (https://www.cosmos.esa.int/gaia)
- **SDSS**: Sloan Digital Sky Survey (https://www.sdss.org/)
- **2MASS**: Two Micron All Sky Survey (https://www.ipac.caltech.edu/2mass/)

Please cite the original surveys when using their data.
