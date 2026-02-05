"""Enhanced Exoplanet preprocessor with host star coordinate integration.

Handles NASA Exoplanet Archive data preprocessing with intelligent coordinate resolution
using Gaia DR3 host star data when direct coordinates are unavailable.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from astro_lab.data.transforms.astronomical import spherical_to_cartesian

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class ExoplanetPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Enhanced preprocessor for exoplanet data with host star coordinate integration.

    Uses real Gaia DR3 data to resolve host star coordinates whenever possible.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize exoplanet preprocessor."""
        default_config = {
            "distance_limit_pc": 500.0,
            "require_mass": False,
            "require_radius": False,
            "require_coordinates": False,  # Make coordinates optional
            "use_gaia_host_coords": True,  # Try to get host star coordinates from Gaia
            "gaia_data_path": None,  # Path to Gaia data for host star lookup
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        # Make required columns more flexible
        self.required_columns = ["pl_name"]  # Only planet name is truly required
        self._gaia_host_data = None  # Cache for Gaia host star data

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "exoplanet"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "exoplanet"

    def _load_gaia_host_data(self) -> Optional[pl.DataFrame]:
        """Load processed Gaia data for host star coordinate lookup."""
        if self._gaia_host_data is not None:
            return self._gaia_host_data

        # Try to find Gaia data in processed directory
        gaia_files = [
            Path("data/processed/gaia/gaia_processed.parquet"),
            Path("data/raw/gaia/gaia_sample.parquet"),
            Path("data/raw/gaia.parquet"),
        ]

        for gaia_path in gaia_files:
            if gaia_path.exists():
                try:
                    logger.info(f"Loading Gaia host star data from {gaia_path}")
                    gaia_data = pl.read_parquet(gaia_path)

                    # Create host star mapping using star names/identifiers
                    # Gaia data might have star names in various columns
                    potential_name_cols = [
                        "source_id",
                        "name",
                        "designation",
                        "dr2_source_id",
                    ]

                    for col in potential_name_cols:
                        if col in gaia_data.columns:
                            # Create a mapping from star names to coordinates
                            self._gaia_host_data = gaia_data.select(
                                [col, "ra", "dec", "distance_pc"]
                            ).rename({col: "host_name"})
                            logger.info(
                                f"Created Gaia host star mapping with {len(self._gaia_host_data)} stars"
                            )
                            return self._gaia_host_data

                except Exception as e:
                    logger.warning(f"Failed to load Gaia data from {gaia_path}: {e}")
                    continue

        logger.warning("No Gaia host star data available for coordinate lookup")
        return None

    def _match_host_stars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Join exoplanet data with Gaia host star data to get coordinates and distance."""
        gaia_hosts = self._load_gaia_host_data()
        if gaia_hosts is None:
            logger.warning("No Gaia host star data available")
            return df

        # Try to match host stars using various column combinations
        host_cols = ["hostname", "pl_hostname", "host_name", "star_name"]
        host_col = None

        for col in host_cols:
            if col in df.columns:
                host_col = col
                break

        if host_col is None:
            logger.warning("No host star name column found in exoplanet data")
            return df

        # Perform the join
        try:
            joined = df.join(
                gaia_hosts.select(["host_name", "ra", "dec", "distance_pc"]),
                left_on=host_col,
                right_on="host_name",
                how="left",
                suffix="_gaia",
            )

            # Use Gaia coordinates if available, otherwise keep original
            for coord_col in ["ra", "dec", "distance_pc"]:
                gaia_col = f"{coord_col}_gaia"
                if gaia_col in joined.columns:
                    # Create a combined column using Gaia data preferentially
                    if coord_col in joined.columns:
                        joined = joined.with_columns(
                            [
                                pl.when(pl.col(gaia_col).is_not_null())
                                .then(pl.col(gaia_col))
                                .otherwise(pl.col(coord_col))
                                .alias(coord_col)
                            ]
                        )
                    else:
                        joined = joined.with_columns(
                            [pl.col(gaia_col).alias(coord_col)]
                        )

                    # Clean up temporary column
                    joined = joined.drop(gaia_col)

            # Count successful matches
            matched_count = joined.filter(
                pl.col("ra").is_not_null() & pl.col("dec").is_not_null()
            ).height

            total_count = len(df)
            match_rate = matched_count / total_count if total_count > 0 else 0

            logger.info(
                f"Matched {matched_count}/{total_count} exoplanets "
                f"({match_rate:.1%}) to Gaia host stars"
            )

            return joined

        except Exception as e:
            logger.error(f"Failed to match exoplanets with Gaia host stars: {e}")
            return df

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply exoplanet-specific filters."""
        initial_count = len(df)

        # Distance filter (try multiple column names)
        distance_cols = ["sy_dist", "st_dist", "distance", "distance_pc"]
        distance_col = None
        for col in distance_cols:
            if col in df.columns:
                distance_col = col
                break

        if distance_col and self.config["distance_limit_pc"]:
            df = df.filter(
                (pl.col(distance_col) > 0)
                & (pl.col(distance_col) < self.config["distance_limit_pc"])
            )

        # Require planet mass
        mass_cols = ["pl_bmasse", "pl_masse", "planet_mass"]
        if self.config["require_mass"]:
            for col in mass_cols:
                if col in df.columns:
                    df = df.filter(pl.col(col).is_not_null())
                    break

        # Require planet radius
        radius_cols = ["pl_rade", "pl_radius", "planet_radius"]
        if self.config["require_radius"]:
            for col in radius_cols:
                if col in df.columns:
                    df = df.filter(pl.col(col).is_not_null())
                    break

        final_count = len(df)
        filtered_count = initial_count - final_count
        logger.info(
            f"Filtered {filtered_count} exoplanets ({filtered_count / initial_count * 100:.0f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform exoplanet data: derive host star coordinates from Gaia when possible."""
        logger.info("Transforming exoplanet data...")

        # Always try to match with Gaia host star data first
        if self.config["use_gaia_host_coords"]:
            df = self._match_host_stars(df)

        # Check if we have coordinates after Gaia matching
        has_coords = all(col in df.columns for col in ["ra", "dec"])

        # If still no coordinates, try to find coordinate columns with various names
        if not has_coords:
            ra_cols = ["ra", "RA", "ra_deg", "RA_deg", "st_ra", "star_ra"]
            dec_cols = ["dec", "DEC", "dec_deg", "DEC_deg", "st_dec", "star_dec"]

            ra_col = None
            dec_col = None

            for col in ra_cols:
                if col in df.columns:
                    ra_col = col
                    break

            for col in dec_cols:
                if col in df.columns:
                    dec_col = col
                    break

            if ra_col and dec_col:
                # Rename to standard names
                df = df.rename({ra_col: "ra", dec_col: "dec"})
                has_coords = True

        # Handle distance information
        if "distance_pc" not in df.columns:
            distance_cols = ["sy_dist", "st_dist", "distance"]
            for col in distance_cols:
                if col in df.columns:
                    df = df.rename({col: "distance_pc"})
                    break

        # Convert to cartesian coordinates if we have all components
        if has_coords and "distance_pc" in df.columns:
            logger.info("Converting host star coordinates to Cartesian")
            x, y, z = spherical_to_cartesian(df["ra"], df["dec"], df["distance_pc"])
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )
        else:
            logger.warning(
                "Insufficient coordinate information found. "
                "Coordinates may be missing for some exoplanets."
            )

        # Calculate planet properties
        df = self._calculate_planet_properties(df)

        # Calculate host star properties
        df = self._calculate_host_properties(df)

        # Calculate planet cartesian coordinates relative to host star
        df = self._calculate_planet_cartesian_coords(df)

        return df

    def _calculate_planet_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate derived planet properties from real data."""
        # Planet mass and radius
        mass_cols = ["pl_bmasse", "pl_masse", "planet_mass"]
        radius_cols = ["pl_rade", "pl_radius", "planet_radius"]

        mass_col = None
        radius_col = None

        for col in mass_cols:
            if col in df.columns:
                mass_col = col
                break

        for col in radius_cols:
            if col in df.columns:
                radius_col = col
                break

        if mass_col and radius_col:
            # Bulk density (Earth units)
            df = df.with_columns(
                [(pl.col(mass_col) / (pl.col(radius_col) ** 3)).alias("pl_density")]
            )

            # Planet type classification based on mass and radius
            df = df.with_columns(
                [
                    pl.when((pl.col(radius_col) < 1.25) & (pl.col(mass_col) < 2.0))
                    .then(pl.lit("terrestrial"))
                    .when((pl.col(radius_col) < 2.0) & (pl.col(mass_col) < 10.0))
                    .then(pl.lit("super_earth"))
                    .when((pl.col(radius_col) < 4.0) & (pl.col(mass_col) < 50.0))
                    .then(pl.lit("neptune"))
                    .otherwise(pl.lit("gas_giant"))
                    .alias("pl_type")
                ]
            )

        # Orbital properties
        period_cols = ["pl_orbper", "period", "orbital_period"]
        period_col = None
        for col in period_cols:
            if col in df.columns:
                period_col = col
                break

        if period_col:
            # Habitability metrics (simplified)
            df = df.with_columns(
                [
                    # Earth-like period range (200-500 days)
                    pl.when((pl.col(period_col) > 200) & (pl.col(period_col) < 500))
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias("potentially_habitable_period")
                ]
            )

        return df

    def _calculate_host_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate host star properties from real data."""
        # Host star temperature
        teff_cols = ["st_teff", "teff", "star_teff", "host_teff"]
        teff_col = None
        for col in teff_cols:
            if col in df.columns:
                teff_col = col
                break

        if teff_col:
            # Stellar type classification (simplified)
            df = df.with_columns(
                [
                    pl.when(pl.col(teff_col) > 7500)
                    .then(pl.lit("A"))
                    .when(pl.col(teff_col) > 6000)
                    .then(pl.lit("F"))
                    .when(pl.col(teff_col) > 5200)
                    .then(pl.lit("G"))
                    .when(pl.col(teff_col) > 3700)
                    .then(pl.lit("K"))
                    .otherwise(pl.lit("M"))
                    .alias("st_type"),
                    # Sun-like stars
                    pl.when((pl.col(teff_col) > 5000) & (pl.col(teff_col) < 6000))
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias("sun_like_host"),
                ]
            )

        # Host star mass
        mass_cols = ["st_mass", "star_mass", "host_mass"]
        mass_col = None
        for col in mass_cols:
            if col in df.columns:
                mass_col = col
                break

        if mass_col:
            df = df.with_columns([pl.col(mass_col).alias("host_mass_solar")])

        return df

    def _calculate_planet_cartesian_coords(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate planet positions relative to host star using real orbital data."""
        # Required columns: host x, y, z and planet orbital elements
        if not all(col in df.columns for col in ["x", "y", "z"]):
            logger.warning(
                "No host star coordinates available for planet position calculation"
            )
            df = df.with_columns(
                [
                    pl.lit(None).alias("planet_x"),
                    pl.lit(None).alias("planet_y"),
                    pl.lit(None).alias("planet_z"),
                ]
            )
            return df

        if "pl_orbsmax" not in df.columns:
            logger.warning("No orbital semi-major axis data available")
            df = df.with_columns(
                [
                    pl.lit(None).alias("planet_x"),
                    pl.lit(None).alias("planet_y"),
                    pl.lit(None).alias("planet_z"),
                ]
            )
            return df

        # Constants
        AU_PC = 4.84813681109536e-6  # 1 AU in parsec

        # Prepare orbital data
        x_s = df["x"].to_numpy()
        y_s = df["y"].to_numpy()
        z_s = df["z"].to_numpy()
        a = df["pl_orbsmax"].to_numpy() * AU_PC  # convert AU to pc

        # Orbital elements (use real data if available, otherwise defaults)
        e = (
            df["pl_orbeccen"].to_numpy()
            if "pl_orbeccen" in df.columns
            else np.zeros_like(a)
        )

        # Orbital phase - use real data if available
        if "pl_orbphase" in df.columns:
            phase = df["pl_orbphase"].to_numpy()
        elif "pl_orbtper" in df.columns and "pl_orbper" in df.columns:
            # Calculate phase from time of periastron and period
            phase = np.mod(
                df["pl_orbtper"].to_numpy() / df["pl_orbper"].to_numpy(), 1.0
            )
        else:
            # Random phase for each planet
            phase = np.random.uniform(0, 1, size=len(a))

        incl = (
            df["pl_orbincl"].to_numpy()
            if "pl_orbincl" in df.columns
            else np.random.uniform(0, 180, size=len(a))  # Random inclinations
        )

        # Calculate position in orbital plane
        theta = 2 * np.pi * phase  # phase in [0,1] -> angle
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # Apply inclination
        incl_rad = np.deg2rad(incl)
        x_orb_incl = x_orb
        y_orb_incl = y_orb * np.cos(incl_rad)
        z_orb_incl = y_orb * np.sin(incl_rad)

        # Add to host star position
        planet_x = x_s + x_orb_incl
        planet_y = y_s + y_orb_incl
        planet_z = z_s + z_orb_incl

        df = df.with_columns(
            [
                pl.Series("planet_x", planet_x),
                pl.Series("planet_y", planet_y),
                pl.Series("planet_z", planet_z),
            ]
        )

        logger.info("Calculated planet Cartesian coordinates relative to host stars")
        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract exoplanet features for ML."""
        logger.info("Extracting features from exoplanet data...")

        feature_columns = []

        # Position (host star coordinates)
        coord_cols = ["x", "y", "z", "ra", "dec", "distance_pc"]
        for col in coord_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Planet coordinates relative to host
        planet_coord_cols = ["planet_x", "planet_y", "planet_z"]
        for col in planet_coord_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Planet properties
        planet_props = {
            "period": ["pl_orbper", "period", "orbital_period"],
            "mass": ["pl_bmasse", "pl_masse", "planet_mass"],
            "radius": ["pl_rade", "pl_radius", "planet_radius"],
            "density": ["pl_density"],
            "type": ["pl_type"],
            "habitable": ["potentially_habitable_period"],
            "semi_major_axis": ["pl_orbsmax"],
            "eccentricity": ["pl_orbeccen"],
            "inclination": ["pl_orbincl"],
        }

        for prop_name, col_names in planet_props.items():
            for col in col_names:
                if col in df.columns:
                    feature_columns.append(col)
                    break

        # Host star properties
        host_props = {
            "teff": ["st_teff", "teff", "star_teff", "host_teff"],
            "mass": ["st_mass", "star_mass", "host_mass_solar"],
            "radius": ["st_rad", "star_radius"],
            "type": ["st_type"],
            "sun_like": ["sun_like_host"],
        }

        for prop_name, col_names in host_props.items():
            for col in col_names:
                if col in df.columns:
                    feature_columns.append(col)
                    break

        # Planet identifier
        id_cols = ["pl_name", "planet_name", "name"]
        for col in id_cols:
            if col in df.columns:
                feature_columns.append(col)
                break

        # Remove duplicates and select available features
        feature_columns = list(set(feature_columns))
        available_features = [col for col in feature_columns if col in df.columns]

        if available_features:
            df = df.select(available_features)
        else:
            logger.warning("No features found, keeping all columns")

        logger.info(
            f"Extracted {len(available_features)} features for {len(df)} exoplanets"
        )

        return df

    def get_brightness_measure(self, df: pl.DataFrame) -> pl.Series:
        """Get brightness measure for visualization."""
        # For exoplanets, use planet radius or mass as brightness proxy
        if "pl_rade" in df.columns:
            return df["pl_rade"]  # Larger planets are "brighter"
        elif "pl_masse" in df.columns:
            return df["pl_masse"]  # More massive planets are "brighter"
        else:
            return pl.Series([1.0] * len(df))
