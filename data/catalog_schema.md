# AstroLab Catalog Schema Documentation

## Overview

The AstroLab catalog (`astrolab_catalog_v1.parquet`) is a consolidated multi-survey catalog combining astronomical data from Gaia, SDSS, and 2MASS with pre-computed cosmic web structure classifications.

## Column Groups

### Coordinate Columns

#### Spherical Coordinates
- `ra` (float64): Right Ascension in degrees (ICRS)
- `dec` (float64): Declination in degrees (ICRS)
- `distance_pc` (float64): Distance in parsecs

#### Cartesian Coordinates
Survey-specific 3D coordinates in parsecs:
- `gaia_x`, `gaia_y`, `gaia_z` (float64): Gaia-derived Cartesian coordinates
- `sdss_x`, `sdss_y`, `sdss_z` (float64): SDSS-derived Cartesian coordinates
- `twomass_x`, `twomass_y`, `twomass_z` (float64): 2MASS-derived Cartesian coordinates

Or generic coordinates if single survey:
- `x`, `y`, `z` (float64): Cartesian coordinates in parsecs

### Astrometry (Gaia)
- `gaia_parallax` (float64): Parallax in milliarcseconds
- `gaia_pmra` (float64): Proper motion in RA (mas/yr)
- `gaia_pmdec` (float64): Proper motion in Dec (mas/yr)
- `gaia_source_id` (int64): Gaia DR3 source identifier

### Photometry

#### Gaia (Optical)
- `gaia_phot_g_mean_mag` (float64): G-band magnitude
- `gaia_phot_bp_mean_mag` (float64): BP (blue) magnitude
- `gaia_phot_rp_mean_mag` (float64): RP (red) magnitude
- `gaia_bp_rp` (float64): BP-RP color index

#### SDSS (Optical)
- `sdss_u` (float64): u-band magnitude
- `sdss_g` (float64): g-band magnitude
- `sdss_r` (float64): r-band magnitude
- `sdss_i` (float64): i-band magnitude
- `sdss_z` (float64): z-band magnitude

#### 2MASS (Near-Infrared)
- `twomass_j_m` (float64): J-band magnitude
- `twomass_h_m` (float64): H-band magnitude
- `twomass_k_m` (float64): K-band magnitude

### Cosmic Web Features

For each clustering scale (e.g., 5.0, 10.0, 25.0, 50.0 pc):

#### Structure Classification
- `cosmic_web_class_{scale}pc` (int32): Structure type classification
  - `0` = **Field**: Low-density background regions
  - `1` = **Filament**: Elongated structures connecting nodes
  - `2` = **Void**: Underdense regions
  - `3` = **Node**: High-density clusters/nodes

#### Density Field
- `density_{scale}pc` (float64): Local density estimate
  - Computed using graph-based neighbor counting
  - Higher values indicate denser regions

#### Structure Metrics
- `anisotropy_{scale}pc` (float64): Structural anisotropy measure (0-1)
  - Values near 0: isotropic (sphere-like)
  - Values near 1: highly anisotropic (filament-like)
  - Computed from eigenvalue ratios of local structure tensor

### Metadata
- `catalog_version` (string): Catalog version (e.g., "v1.0")
- `processing_date` (string): ISO 8601 timestamp of catalog generation

## Usage Examples

### Basic Filtering

```python
import polars as pl

# Load catalog
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")

# Get filament structures at 10 pc scale
filaments = catalog.filter(pl.col("cosmic_web_class_10.0pc") == 1)

# Get high-density regions
density_threshold = catalog["density_10.0pc"].quantile(0.9)
high_density = catalog.filter(pl.col("density_10.0pc") > density_threshold)

# Get nodes (clusters) at 25 pc scale
nodes = catalog.filter(pl.col("cosmic_web_class_25.0pc") == 3)
```

### Multi-Scale Analysis

```python
# Compare structure classification across scales
scales = [5.0, 10.0, 25.0, 50.0]

for scale in scales:
    class_col = f"cosmic_web_class_{scale}pc"
    counts = catalog[class_col].value_counts()
    print(f"\nScale {scale} pc:")
    print(counts)
```

### Photometric Selection

```python
# Red stars (potential red giants)
red_stars = catalog.filter(
    (pl.col("gaia_bp_rp") > 1.0) &  # Red color
    (pl.col("gaia_phot_g_mean_mag") < 15)  # Bright
)

# Cross-matched sources (all three surveys)
complete = catalog.filter(
    pl.col("gaia_phot_g_mean_mag").is_not_null() &
    pl.col("sdss_r").is_not_null() &
    pl.col("twomass_k_m").is_not_null()
)
```

### Combining Filters

```python
# Dense filamentary structures at 10 pc with red colors
dense_red_filaments = catalog.filter(
    (pl.col("cosmic_web_class_10.0pc") == 1) &  # Filaments
    (pl.col("density_10.0pc") > density_threshold) &  # High density
    (pl.col("gaia_bp_rp") > 1.0)  # Red color
)

# Nearby sources in voids
nearby_voids = catalog.filter(
    (pl.col("cosmic_web_class_25.0pc") == 2) &  # Void
    (pl.col("distance_pc") < 100)  # < 100 pc
)
```

### Extracting Coordinates

```python
# Get 3D coordinates for plotting
coords = catalog.select(['x', 'y', 'z']).to_numpy()

# Or survey-specific coordinates
gaia_coords = catalog.select(['gaia_x', 'gaia_y', 'gaia_z']).to_numpy()
```

## Data Quality Notes

### Missing Values
- Not all sources have measurements in all surveys
- Cross-matching uses 1 arcsec radius by default
- Use `.is_not_null()` to filter complete records

### Scale Selection
- Smaller scales (5-10 pc): Local stellar structures
- Medium scales (10-25 pc): Stellar associations
- Larger scales (25-50 pc): Extended structures

### Coordinate Systems
- All coordinates are in ICRS (International Celestial Reference System)
- Distances derived from Gaia parallaxes (when available)
- Cartesian coordinates computed from (RA, Dec, distance)

## Version History

### v1.0 (2026-02)
- Initial release
- Gaia DR3, SDSS, 2MASS integration
- Multi-scale cosmic web analysis
- Structure classification (field, filament, void, node)
- Density and anisotropy metrics

## Citation

If you use this catalog in your research, please cite:

```bibtex
@software{astrolab_catalog,
  title = {AstroLab Catalog: Multi-Survey Cosmic Web Dataset},
  author = {AstroLab Team},
  year = {2026},
  url = {https://github.com/synapticore-io/astro-lab},
  version = {1.0}
}
```

And cite the original surveys:
- **Gaia**: [Gaia Collaboration et al. (2016, 2023)](https://www.cosmos.esa.int/gaia)
- **SDSS**: [York et al. (2000)](https://www.sdss.org/)
- **2MASS**: [Skrutskie et al. (2006)](https://www.ipac.caltech.edu/2mass/)
