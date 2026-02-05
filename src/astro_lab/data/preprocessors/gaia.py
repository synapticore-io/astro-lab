"""Enhanced Gaia survey preprocessor implementation with unified 3D coordinate handling.

Handles Gaia DR3 data preprocessing for machine learning applications with
automatic 3D coordinate conversion and standardized field mapping.
"""

import logging
from typing import Any, Dict, List, Optional

import polars as pl

from astro_lab.config import get_survey_config

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class GaiaPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Enhanced preprocessor for Gaia DR3 data with unified 3D coordinate handling.

    Handles:
    - Quality filtering (parallax SNR, excess noise, RUWE)
    - Automatic 3D coordinate conversion (spherical → Cartesian)
    - Feature extraction (colors, kinematics, stellar properties)
    - Unified field mapping for spatial analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Gaia preprocessor.

        Args:
            config: Configuration dict with preprocessing parameters
        """
        default_config = {
            "parallax_snr_min": None,
            "astrometric_excess_noise_max": None,
            "ruwe_max": None,
            "magnitude_limit": None,
            "distance_limit_pc": None,
            "remove_duplicates": False,
            "handle_missing": "median",
            "include_galactic_coords": True,
            "include_kinematics": True,
            "include_colors": True,
            "map_columns": True,  # Enable column mapping
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        # Get survey configuration
        self.survey_config = get_survey_config("gaia")

        # Column mapping for Gaia (handles various formats)
        self.column_mapping = {
            # Standard Gaia columns
            "SOURCE_ID": "source_id",
            "RA_ICRS": "ra",
            "DE_ICRS": "dec",
            "Plx": "parallax",
            "e_Plx": "parallax_error",
            "pmRA": "pmra",
            "pmDE": "pmdec",
            "e_pmRA": "pmra_error",
            "e_pmDE": "pmdec_error",
            "Gmag": "phot_g_mean_mag",
            "BPmag": "phot_bp_mean_mag",
            "RPmag": "phot_rp_mean_mag",
            "RV": "radial_velocity",
            "e_RV": "radial_velocity_error",
            # Alternative column names
            "source_id": "source_id",
            "ra": "ra",
            "dec": "dec",
            "parallax": "parallax",
            "parallax_error": "parallax_error",
            "pmra": "pmra",
            "pmdec": "pmdec",
            "pmra_error": "pmra_error",
            "pmdec_error": "pmdec_error",
            "phot_g_mean_mag": "phot_g_mean_mag",
            "phot_bp_mean_mag": "phot_bp_mean_mag",
            "phot_rp_mean_mag": "phot_rp_mean_mag",
            "radial_velocity": "radial_velocity",
            "radial_velocity_error": "radial_velocity_error",
            # Quality columns
            "astrometric_excess_noise": "astrometric_excess_noise",
            "ruwe": "ruwe",
            "visibility_periods_used": "visibility_periods_used",
        }

        # Gaia-specific column sets
        self.required_columns = [
            "source_id",
            "ra",
            "dec",
            "parallax",
            "parallax_error",
            "pmra",
            "pmdec",
            "phot_g_mean_mag",
        ]

        self.quality_columns = [
            "astrometric_excess_noise",
            "ruwe",
            "visibility_periods_used",
        ]

        self.photometry_columns = [
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
        ]

        self.kinematics_columns = [
            "pmra",
            "pmdec",
            "pmra_error",
            "pmdec_error",
            "radial_velocity",
            "radial_velocity_error",
        ]

    def get_survey_name(self) -> str:
        """Get the survey name for this preprocessor."""
        return "gaia"

    def get_object_type(self) -> str:
        """Get the primary object type for this survey."""
        return "star"

    def _map_and_select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map column names and select relevant columns for Gaia."""
        if not self.config.get("map_columns", True):
            return df

        logger.info("Mapping and selecting columns for gaia")

        # Apply column mapping (case-insensitive)
        available_columns = df.columns
        mapping_applied = {}

        for old_name, new_name in self.column_mapping.items():
            if old_name in available_columns:
                mapping_applied[old_name] = new_name
            else:
                for col in available_columns:
                    if col.lower() == old_name.lower():
                        mapping_applied[col] = new_name
                        break

        # Apply mapping
        if mapping_applied:
            df = df.rename(mapping_applied)
            logger.info(f"Applied column mapping: {mapping_applied}")

        # Select essential and optional columns for Gaia
        essential_columns = ["source_id", "ra", "dec", "parallax"]
        optional_columns = [
            "parallax_error",
            "pmra",
            "pmdec",
            "pmra_error",
            "pmdec_error",
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "radial_velocity",
            "radial_velocity_error",
            "astrometric_excess_noise",
            "ruwe",
            "visibility_periods_used",
        ]

        # Keep columns that exist
        columns_to_keep = []
        missing = []
        for col in essential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
            else:
                found = False
                for c in df.columns:
                    if c.lower() == col.lower():
                        columns_to_keep.append(c)
                        found = True
                        break
                if not found:
                    missing.append(col)

        for col in optional_columns:
            if col in df.columns:
                columns_to_keep.append(col)

        # Remove duplicates
        columns_to_keep = list(dict.fromkeys(columns_to_keep))

        if missing:
            logger.error(f"Missing essential columns after mapping: {missing}")
            logger.error(f"Available columns: {df.columns}")
            raise ValueError(
                f"Missing essential columns in Gaia data: {missing}\nAvailable columns: {list(df.columns)}"
            )

        df = df.select(columns_to_keep)
        logger.info(f"Selected {len(columns_to_keep)} columns for Gaia processing")
        return df

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Gaia-specific quality filters.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        logger.info("Applying Gaia quality filters...")

        # Apply column mapping first
        df = self._map_and_select_columns(df)

        initial_count = len(df)

        # Basic quality filters
        filters = []

        # Parallax SNR filter
        if (
            "parallax" in df.columns
            and "parallax_error" in df.columns
            and self.config["parallax_snr_min"] is not None
        ):
            parallax_snr = df["parallax"] / df["parallax_error"]
            filters.append(parallax_snr > self.config["parallax_snr_min"])
            filters.append(df["parallax"] > 0)  # Positive parallax only
            logger.info(
                f"Applied parallax SNR filter: min_snr={self.config['parallax_snr_min']}"
            )

        # Astrometric excess noise
        if (
            "astrometric_excess_noise" in df.columns
            and self.config["astrometric_excess_noise_max"] is not None
        ):
            filters.append(
                df["astrometric_excess_noise"]
                < self.config["astrometric_excess_noise_max"]
            )
            logger.info(
                f"Applied astrometric excess noise filter: max={self.config['astrometric_excess_noise_max']}"
            )

        # RUWE (Renormalized Unit Weight Error)
        if "ruwe" in df.columns and self.config["ruwe_max"] is not None:
            filters.append(df["ruwe"] < self.config["ruwe_max"])
            logger.info(f"Applied RUWE filter: max={self.config['ruwe_max']}")

        # Magnitude limit
        if (
            "phot_g_mean_mag" in df.columns
            and self.config["magnitude_limit"] is not None
        ):
            filters.append(df["phot_g_mean_mag"] < self.config["magnitude_limit"])
            logger.info(
                f"Applied magnitude limit: max={self.config['magnitude_limit']}"
            )

        # Apply all filters
        if filters:
            mask = filters[0]
            for f in filters[1:]:
                mask = mask & f
            df = df.filter(mask)

        # Distance filter (if parallax available)
        if "parallax" in df.columns and self.config["distance_limit_pc"] is not None:
            distance_pc = 1000.0 / df["parallax"]  # Convert mas to pc
            df = df.filter(distance_pc < self.config["distance_limit_pc"])
            logger.info(
                f"Applied distance limit: max={self.config['distance_limit_pc']} pc"
            )

        # Remove duplicates by source_id
        if self.config["remove_duplicates"] and "source_id" in df.columns:
            df = df.unique(subset=["source_id"])
            logger.info("Removed duplicate source_id entries")

        final_count = len(df)
        self.stats["filter"] = {
            "initial_count": initial_count,
            "final_count": final_count,
            "filtered_fraction": 1 - (final_count / initial_count),
        }

        logger.info(
            f"Filtered {initial_count - final_count} objects "
            f"({100 * (1 - final_count / initial_count):.1f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform Gaia data for unified 3D processing.

        Includes:
        - Parallax to distance conversion
        - Galactic coordinate calculation
        - Color index calculation
        - Kinematic feature calculation
        - Missing value handling

        Args:
            df: Filtered DataFrame

        Returns:
            Transformed DataFrame with standardized coordinates
        """
        logger.info("Transforming Gaia data...")

        # Convert parallax to distance (this will be used by ensure_3d_coordinates)
        if "parallax" in df.columns and "distance_pc" not in df.columns:
            df = df.with_columns(
                [
                    pl.when(pl.col("parallax") > 0.001)
                    .then(1000.0 / pl.col("parallax"))
                    .otherwise(pl.lit(None))
                    .alias("distance_pc")
                ]
            )
            logger.info("Converted parallax to distance")

        # Add galactic coordinates if requested
        if self.config["include_galactic_coords"]:
            df = self._add_galactic_coordinates(df)
            logger.info("Added galactic coordinates")

        # Calculate colors if photometry available
        if self.config["include_colors"] and all(
            col in df.columns for col in ["phot_bp_mean_mag", "phot_rp_mean_mag"]
        ):
            df = df.with_columns(
                [
                    (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias(
                        "bp_rp"
                    ),
                    (pl.col("phot_g_mean_mag") - pl.col("phot_rp_mean_mag")).alias(
                        "g_rp"
                    ),
                    (pl.col("phot_g_mean_mag") - pl.col("phot_bp_mean_mag")).alias(
                        "g_bp"
                    ),
                ]
            )
            logger.info("Calculated color indices")

        # Calculate kinematic features
        if self.config["include_kinematics"] and all(
            col in df.columns for col in ["pmra", "pmdec"]
        ):
            df = df.with_columns(
                [(pl.col("pmra") ** 2 + pl.col("pmdec") ** 2).sqrt().alias("pm_total")]
            )

            # Calculate tangential velocity (km/s) if distance available
            if "distance_pc" in df.columns:
                df = df.with_columns(
                    [
                        (
                            4.74 * pl.col("pm_total") * pl.col("distance_pc") / 1000
                        ).alias("v_tan_kms")
                    ]
                )
                logger.info("Calculated kinematic features")

        # Calculate stellar properties
        df = self._calculate_stellar_properties(df)

        # Add cartesian coordinates if not already present and if possible
        if all(col in df.columns for col in ["ra", "dec", "distance_pc"]) and not all(
            col in df.columns for col in ["x", "y", "z"]
        ):
            from astro_lab.data.transforms.astronomical import spherical_to_cartesian

            x, y, z = spherical_to_cartesian(df["ra"], df["dec"], df["distance_pc"])
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )

        self.stats["transform"] = {
            "num_features": len(df.columns),
            "has_colors": "bp_rp" in df.columns,
            "has_kinematics": "v_tan_kms" in df.columns,
            "has_galactic_coords": "gal_l" in df.columns,
        }

        return df

    def _add_galactic_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add galactic coordinates using astropy."""
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        # Get RA/Dec values
        ra_vals = df["ra"].to_numpy()
        dec_vals = df["dec"].to_numpy()
        degree = u.Unit("deg")

        # Convert to galactic coordinates
        coords = SkyCoord(ra=ra_vals * degree, dec=dec_vals * degree, frame="icrs")
        galactic = coords.galactic

        return df.with_columns(
            [
                pl.Series("gal_l", galactic.l.degree),
                pl.Series("gal_b", galactic.b.degree),
            ]
        )

    def _calculate_stellar_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate stellar properties from Gaia data."""
        # Absolute magnitude if distance available
        if "distance_pc" in df.columns and "phot_g_mean_mag" in df.columns:
            df = df.with_columns(
                [
                    (
                        pl.col("phot_g_mean_mag")
                        - 5 * (pl.col("distance_pc").log10() - 1)
                    ).alias("mg_abs")
                ]
            )

        # Stellar mass estimation (very rough approximation)
        if "mg_abs" in df.columns and "bp_rp" in df.columns:
            # Simple mass-luminosity relation approximation
            df = df.with_columns(
                [
                    pl.when(pl.col("mg_abs") < 4.83)  # Brighter than Sun
                    .then(
                        pl.lit(1.0) * (10 ** (-0.4 * (pl.col("mg_abs") - 4.83)) * 0.23)
                    )
                    .otherwise(
                        pl.lit(0.1) * (10 ** (-0.4 * (pl.col("mg_abs") - 4.83)) * 0.23)
                    )
                    .alias("stellar_mass_est")
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract ML-ready features from Gaia data keeping essential coordinates.

        Args:
            df: Transformed DataFrame

        Returns:
            DataFrame with extracted features optimized for ML
        """
        logger.info("Extracting ML features from Gaia data...")

        feature_columns = []

        # Essential identifier
        if "source_id" in df.columns:
            feature_columns.append("source_id")

        # ALWAYS keep coordinate features for 3D processing
        essential_coords = ["ra", "dec"]
        for col in essential_coords:
            if col in df.columns:
                feature_columns.append(col)

        # Distance information
        distance_cols = ["distance_pc", "parallax"]
        for col in distance_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Galactic coordinates
        if self.config["include_galactic_coords"]:
            gal_cols = ["gal_l", "gal_b"]
            for col in gal_cols:
                if col in df.columns:
                    feature_columns.append(col)

        # Photometric features
        photo_cols = ["phot_g_mean_mag", "phot_g_mean_mag_error"]
        if self.config["include_colors"]:
            photo_cols.extend(
                ["phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "g_rp", "g_bp"]
            )
        for col in photo_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Kinematic features
        if self.config["include_kinematics"]:
            kinematic_cols = ["pmra", "pmdec", "pm_total", "v_tan_kms"]
            if "radial_velocity" in df.columns:
                kinematic_cols.append("radial_velocity")
            for col in kinematic_cols:
                if col in df.columns:
                    feature_columns.append(col)

        # Stellar properties
        stellar_cols = ["mg_abs", "stellar_mass_est"]
        for col in stellar_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Quality indicators
        quality_cols = ["parallax_error", "pmra_error", "pmdec_error", "ruwe"]
        for col in quality_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Astrometric quality
        astrometric_cols = ["astrometric_excess_noise", "visibility_periods_used"]
        for col in astrometric_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Remove duplicates and keep only available columns
        feature_columns = list(set(feature_columns))
        available_features = [col for col in feature_columns if col in df.columns]

        # Ensure we have essential coordinates
        if not all(col in available_features for col in ["ra", "dec"]):
            logger.error("Missing essential coordinate columns in feature extraction")
            # Add them back if they exist in the original dataframe
            for col in ["ra", "dec"]:
                if col in df.columns and col not in available_features:
                    available_features.append(col)

        # Select and reorder columns
        df = df.select(available_features)

        # Store feature information
        self.feature_names = available_features

        self.stats["extract_features"] = {
            "num_features": len(available_features),
            "feature_groups": {
                "coordinates": len(
                    [
                        c
                        for c in available_features
                        if c in ["ra", "dec", "gal_l", "gal_b"]
                    ]
                ),
                "photometry": len([c for c in available_features if c in photo_cols]),
                "kinematics": len(
                    [c for c in available_features if c in kinematic_cols]
                ),
                "stellar_properties": len(
                    [c for c in available_features if c in stellar_cols]
                ),
                "quality": len([c for c in available_features if c in quality_cols]),
            },
        }

        logger.info(f"Extracted {len(available_features)} features")
        logger.info(
            f"Feature groups: {self.stats['extract_features']['feature_groups']}"
        )

        return df

    def get_cartesian_columns(self) -> List[str]:
        """Get names of Cartesian coordinate columns."""
        return ["x", "y", "z"]

    def get_feature_columns(self) -> List[str]:
        """Get names of all feature columns."""
        return self.feature_names

    def validate_schema(self, df: pl.DataFrame) -> bool:
        """Validate that DataFrame has required Gaia columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for essential columns after mapping
        essential_cols = ["ra", "dec"]
        missing_columns = [col for col in essential_cols if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing essential columns after mapping: {missing_columns}"
            )

        return True

    def get_stellar_mass(self, df: pl.DataFrame) -> Optional[pl.Series]:
        """Get stellar mass estimates if available."""
        if "stellar_mass_est" in df.columns:
            return df["stellar_mass_est"]
        elif "mass" in df.columns:
            return df["mass"]
        else:
            return None

    def get_brightness_measure(self, df: pl.DataFrame) -> pl.Series:
        """Get brightness measure for visualization."""
        if "phot_g_mean_mag" in df.columns:
            # Convert magnitude to brightness (inverted scale)
            return 25.0 - df["phot_g_mean_mag"]
        else:
            return pl.Series([1.0] * len(df))

    def add_standard_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add standard fields for compatibility with other surveys."""
        # Add object_id if missing
        if "object_id" not in df.columns:
            id_candidates = [
                "source_id",
                "nsaid",
                "objid",
                "id",
                "SOURCE_ID",
                "NSAID",
                "OBJID",
                "ID",
            ]
            id_col = None
            for candidate in id_candidates:
                if candidate in df.columns:
                    id_col = candidate
                    break
            if id_col:
                df = df.with_columns([pl.col(id_col).alias("object_id")])
            else:
                df = df.with_row_count("object_id")
        # Add magnitude if missing
        if "magnitude" not in df.columns:
            mag_cols = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
            for mag_col in mag_cols:
                if mag_col in df.columns:
                    df = df.with_columns([pl.col(mag_col).alias("magnitude")])
                    break
        # Add brightness if missing
        if "brightness" not in df.columns:
            if "magnitude" in df.columns:
                df = df.with_columns([(25.0 - pl.col("magnitude")).alias("brightness")])
            else:
                df = df.with_columns([pl.lit(1.0).alias("brightness")])

        return df
