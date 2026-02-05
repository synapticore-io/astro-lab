"""Enhanced base preprocessor with unified 3D coordinate handling and survey column mapping."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from astropy.cosmology import FlatLambdaCDM

from astro_lab.config import get_data_paths, get_survey_config

logger = logging.getLogger(__name__)

# Default cosmology for distance calculations
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)


class AstroLabDataPreprocessor(ABC):
    """Enhanced base class with unified 3D coordinate handling and survey column mapping.

    This base class ensures all surveys produce real 3D coordinates (x, y, z) for spatial analysis
    and standardizes field mapping across different astronomical surveys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced preprocessor with configuration."""
        self.config = config or {}
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.coordinate_system: str = "cartesian"
        self.distance_unit: str = "pc"
        self.stats: Dict[str, Any] = {}
        self._coordinate_stats: Dict[str, Any] = {}
        self.cosmology = DEFAULT_COSMOLOGY

        # Get survey config for column mapping
        survey_name = self.get_survey_name()
        self.survey_config = get_survey_config(survey_name)

    @abstractmethod
    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter dataframe to remove low-quality or unwanted data."""

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform column values and add derived columns."""

    @abstractmethod
    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract features for machine learning."""

    @abstractmethod
    def get_survey_name(self) -> str:
        """Get the survey name for this preprocessor."""

    @abstractmethod
    def get_object_type(self) -> str:
        """Get the primary object type for this survey."""

    def map_and_select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map survey columns to standard names and select relevant columns."""
        logger.info(f"Mapping and selecting columns for {self.get_survey_name()}")

        # Get survey configuration
        coord_cols = self.survey_config.get("coord_cols", [])
        mag_cols = self.survey_config.get("mag_cols", [])
        extra_cols = self.survey_config.get("extra_cols", [])

        # Create mapping dict and select available columns
        column_mapping = {}
        selected_columns = []

        # Map coordinates
        if len(coord_cols) >= 2:
            ra_col = self._find_column(df, coord_cols[0])
            dec_col = self._find_column(df, coord_cols[1])

            if ra_col:
                column_mapping[ra_col] = "ra"
                selected_columns.append(ra_col)
            if dec_col:
                column_mapping[dec_col] = "dec"
                selected_columns.append(dec_col)

            # Third coordinate (distance/redshift)
            if len(coord_cols) >= 3:
                third_col = self._find_column(df, coord_cols[2])
                if third_col:
                    if coord_cols[2].lower() in ["z", "redshift"]:
                        column_mapping[third_col] = "z"
                    else:
                        column_mapping[third_col] = "distance_pc"
                    selected_columns.append(third_col)

        # Map magnitudes
        for mag_col in mag_cols:
            col = self._find_column(df, mag_col)
            if col:
                # Keep original name but ensure it's selected
                selected_columns.append(col)

        # Map extra columns
        for extra_col in extra_cols:
            col = self._find_column(df, extra_col)
            if col:
                selected_columns.append(col)

        # Add ID columns if available
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
        for id_col in id_candidates:
            if id_col in df.columns and id_col not in selected_columns:
                selected_columns.append(id_col)
                break

        # Select and rename columns
        if selected_columns:
            df_selected = df.select(selected_columns)

            # Apply renaming
            if column_mapping:
                df_selected = df_selected.rename(column_mapping)
                logger.info(f"Applied column mapping: {column_mapping}")
        else:
            logger.warning("No columns found matching survey configuration")
            df_selected = df

        logger.info(
            f"Selected {len(df_selected.columns)} columns: {list(df_selected.columns)[:10]}..."
        )
        return df_selected

    def _find_column(self, df: pl.DataFrame, col_name: str) -> Optional[str]:
        """Find column with case-insensitive matching."""
        # Exact match first
        if col_name in df.columns:
            return col_name

        # Case-insensitive match
        col_lower = col_name.lower()
        for df_col in df.columns:
            if df_col.lower() == col_lower:
                return df_col

        return None

    def ensure_3d_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure that the DataFrame has real 3D spatial coordinates (x, y, z).

        This method ensures all astronomical data has proper 3D Cartesian coordinates
        for spatial analysis and cosmic web processing.

        Returns:
            DataFrame with x, y, z columns added if possible

        Raises:
            ValueError: If coordinates cannot be derived from available data
        """
        # If already present, validate and return
        if all(col in df.columns for col in ["x", "y", "z"]):
            logger.info("3D coordinates already present")
            return df

        # Try to compute from ra, dec, distance
        if all(col in df.columns for col in ["ra", "dec"]):
            if "distance_pc" in df.columns:
                logger.info(
                    "Converting spherical coordinates (ra, dec, distance_pc) to Cartesian"
                )
                from astro_lab.data.transforms.astronomical import (
                    spherical_to_cartesian,
                )

                x, y, z = spherical_to_cartesian(df["ra"], df["dec"], df["distance_pc"])
                df = df.with_columns(
                    [
                        pl.Series("x", x),
                        pl.Series("y", y),
                        pl.Series("z", z),
                    ]
                )

                # Log coordinate statistics
                self._coordinate_stats = {
                    "coordinate_system": "cartesian_from_spherical",
                    "x_range": [float(x.min()), float(x.max())],
                    "y_range": [float(y.min()), float(y.max())],
                    "z_range": [float(z.min()), float(z.max())],
                    "distance_range_pc": [
                        float(df["distance_pc"].min()),
                        float(df["distance_pc"].max()),
                    ],
                }
                logger.info(
                    f"Coordinate conversion successful: {self._coordinate_stats}"
                )
                return df

            elif "distance_mpc" in df.columns:
                logger.info(
                    "Converting spherical coordinates (ra, dec, distance_mpc) to Cartesian"
                )
                from astro_lab.data.transforms.astronomical import (
                    spherical_to_cartesian,
                )

                x, y, z = spherical_to_cartesian(
                    df["ra"], df["dec"], df["distance_mpc"] * 1e6
                )
                df = df.with_columns(
                    [
                        pl.Series("x", x),
                        pl.Series("y", y),
                        pl.Series("z", z),
                    ]
                )
                return df

            elif "z" in df.columns:
                # Convert redshift to distance using cosmology
                logger.info(
                    "Converting redshift to distance for coordinate calculation"
                )
                df = self._redshift_to_distance(df)

                if "distance_pc" in df.columns:
                    from astro_lab.data.transforms.astronomical import (
                        spherical_to_cartesian,
                    )

                    x, y, z = spherical_to_cartesian(
                        df["ra"], df["dec"], df["distance_pc"]
                    )
                    df = df.with_columns(
                        [
                            pl.Series("x", x),
                            pl.Series("y", y),
                            pl.Series("z", z),
                        ]
                    )
                    return df

        # Try parallax conversion for Gaia-like data
        if "parallax" in df.columns:
            logger.info("Converting parallax to distance for coordinate calculation")
            df = self._parallax_to_distance(df)

            if all(col in df.columns for col in ["ra", "dec", "distance_pc"]):
                from astro_lab.data.transforms.astronomical import (
                    spherical_to_cartesian,
                )

                x, y, z = spherical_to_cartesian(df["ra"], df["dec"], df["distance_pc"])
                df = df.with_columns(
                    [
                        pl.Series("x", x),
                        pl.Series("y", y),
                        pl.Series("z", z),
                    ]
                )
                return df

        # If we get here, we couldn't derive 3D coordinates
        logger.error("Cannot derive 3D coordinates from available data")
        logger.error(f"Available columns: {df.columns}")

        raise ValueError(
            f"Cannot derive real 3D coordinates for {self.get_survey_name()} data. "
            f"Available columns: {list(df.columns)}. "
            f"Need either: (x,y,z) or (ra,dec,distance_pc) or (ra,dec,parallax) or (ra,dec,z)"
        )

    def _parallax_to_distance(
        self, df: pl.DataFrame, parallax_col: str = "parallax"
    ) -> pl.DataFrame:
        """Convert parallax to distance in parsecs."""
        df = df.with_columns(
            [
                pl.when(pl.col(parallax_col) > 0.001)
                .then(1000.0 / pl.col(parallax_col))
                .otherwise(pl.lit(None))
                .alias("distance_pc")
            ]
        )

        # Remove objects with invalid distances
        initial_count = len(df)
        df = df.filter(pl.col("distance_pc").is_not_null())
        final_count = len(df)

        if initial_count != final_count:
            logger.warning(
                f"Removed {initial_count - final_count} objects with invalid parallax"
            )

        return df

    def _redshift_to_distance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert redshift to distance using cosmology."""
        redshift_col = "z" if "z" in df.columns else "redshift"

        # Get redshift values
        redshift_values = df[redshift_col].to_numpy()

        # Calculate luminosity distance in Mpc
        distances_mpc = self.cosmology.luminosity_distance(redshift_values).value

        # Convert to parsecs
        distances_pc = distances_mpc * 1e6

        df = df.with_columns([pl.Series("distance_pc", distances_pc)])

        return df

    def add_standard_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add standard fields expected by the unified system."""
        # Add survey name
        df = df.with_columns([pl.lit(self.get_survey_name()).alias("survey_name")])

        # Add object type
        df = df.with_columns([pl.lit(self.get_object_type()).alias("object_type")])

        # Ensure object_id exists
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
                # Create sequential IDs
                df = df.with_row_count("object_id")

        # Ensure magnitude exists - use first available magnitude column
        if "magnitude" not in df.columns:
            mag_cols = self.survey_config.get("mag_cols", [])
            for mag_col in mag_cols:
                if mag_col in df.columns:
                    df = df.with_columns([pl.col(mag_col).alias("magnitude")])
                    break

        # Ensure brightness exists (for visualization)
        if "brightness" not in df.columns:
            if "magnitude" in df.columns:
                # Convert magnitude to brightness (arbitrary scale for visualization)
                df = df.with_columns([(25.0 - pl.col("magnitude")).alias("brightness")])
            else:
                df = df.with_columns([pl.lit(1.0).alias("brightness")])

        return df

    def drop_fully_null_feature_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove only rows where all ML-relevant features are null/NaN/None."""
        feature_cols = (
            self.feature_names
            if self.feature_names
            else [
                col
                for col in df.columns
                if col not in {"object_id", "survey_name", "object_type", "brightness"}
            ]
        )
        if not feature_cols:
            return df
        # Polars-native: keep rows where at least one feature col is not null
        mask = pl.fold(
            acc=pl.lit(False),
            function=lambda acc, s: acc | s.is_not_null(),
            exprs=[pl.col(col) for col in feature_cols],
        )
        filtered = df.filter(mask)
        n_removed = len(df) - len(filtered)
        if n_removed > 0:
            logger.info(
                f"Removed {n_removed} rows where all ML-relevant features are null."
            )
        return filtered

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute complete preprocessing pipeline with unified 3D coordinates."""
        logger.info(f"Starting preprocessing for {self.get_survey_name()} survey")

        # Step 1: Map and select columns according to survey config
        df = self.map_and_select_columns(df)
        logger.info(f"After column mapping: {len(df.columns)} columns")

        # Step 2: Filter data
        df = self.filter(df)
        logger.info(f"After filtering: {len(df)} objects")

        # Step 3: Transform survey-specific data
        df = self.transform(df)
        logger.info(f"After transformation: {len(df.columns)} columns")

        # Step 4: Extract features
        df = self.extract_features(df)
        logger.info(f"After feature extraction: {len(df.columns)} columns")

        # Set feature_names automatically if not set
        if not self.feature_names:
            # Use all float/numeric columns except meta columns
            meta_cols = {"object_id", "survey_name", "object_type", "brightness"}
            self.feature_names = [
                col
                for col, dtype in zip(df.columns, df.dtypes)
                if col not in meta_cols
                and dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]
            logger.info(f"Auto-detected feature columns: {self.feature_names}")
        else:
            logger.info(f"Using configured feature columns: {self.feature_names}")

        # Impute and cast all features robustly
        for col in self.feature_names:
            if col in df.columns:
                n_nan = df[col].is_nan().sum() if hasattr(df[col], "is_nan") else 0
                n_null = df[col].is_null().sum()
                if n_nan > 0 or n_null > 0:
                    logger.info(
                        f"Imputing {n_nan} NaN and {n_null} nulls in feature '{col}' with -999 and casting to float32."
                    )
                df = df.with_columns(
                    pl.col(col).fill_nan(-999).fill_null(-999).cast(pl.Float32)
                )

        # Remove feature columns that are still fully null (all values missing)
        fully_null_features = [
            col
            for col in self.feature_names
            if col in df.columns and df[col].is_null().all()
        ]
        if fully_null_features:
            logger.info(f"Dropping fully-null feature columns: {fully_null_features}")
            df = df.drop(fully_null_features)
            self.feature_names = [
                col for col in self.feature_names if col not in fully_null_features
            ]

        # Log remaining NaN per feature
        for col in self.feature_names:
            if col in df.columns:
                n_remaining = df[col].is_null().sum()
                if n_remaining > 0:
                    logger.warning(
                        f"Feature '{col}' still has {n_remaining} NaN values after imputation! Forcing fill with -999."
                    )
                    df = df.with_columns(pl.col(col).fill_nan(-999).fill_null(-999))

        # Step 5: Ensure unified 3D coordinates
        df = self.ensure_3d_coordinates(df)
        logger.info(f"After 3D coordinate unification: {len(df)} objects")

        # Step 6: Add standard fields
        df = self.add_standard_fields(df)
        logger.info(f"After standardization: {len(df.columns)} columns")

        # Step 7: Remove only rows where all ML-relevant features are null
        df = self.drop_fully_null_feature_rows(df)

        # Update metadata
        self.metadata.update(
            {
                "n_rows": len(df),
                "n_features": len(df.columns),
                "feature_names": df.columns,
                "coordinate_system": self.coordinate_system,
                "distance_unit": self.distance_unit,
                "survey_name": self.get_survey_name(),
                "object_type": self.get_object_type(),
                "coordinate_stats": self._coordinate_stats,
            }
        )

        logger.info(
            f"Preprocessing complete: {len(df)} objects with unified 3D coordinates"
        )

        return df

    def get_info(self) -> Dict[str, Any]:
        """Get preprocessor information and statistics."""
        return {
            "config": self.config,
            "metadata": self.metadata,
            "feature_names": self.feature_names,
            "survey_name": self.get_survey_name(),
            "object_type": self.get_object_type(),
            "survey_config": self.survey_config,
            "coordinate_stats": self._coordinate_stats,
        }

    def _find_data_file(self) -> Path:
        """Return the path to the raw data file for this survey using the central config only."""
        data_dir = get_data_paths()["raw_dir"]
        survey_name = self.get_survey_name()
        survey_dir = Path(data_dir) / survey_name
        for pattern in ["*.parquet", "*.fits"]:
            files = list(survey_dir.glob(pattern))
            if files:
                return files[0]
        return survey_dir / f"{survey_name}.parquet"


class AstronomicalPreprocessorMixin:
    """Mixin providing common astronomical preprocessing utilities."""

    @staticmethod
    def calculate_colors(df: pl.DataFrame, mag_columns: List[str]) -> pl.DataFrame:
        """Calculate color indices from magnitude columns."""
        result_df = df
        for i, mag1 in enumerate(mag_columns):
            for mag2 in mag_columns[i + 1 :]:
                color_name = f"{mag1}_{mag2}_color"
                result_df = result_df.with_columns(
                    [(pl.col(mag1) - pl.col(mag2)).alias(color_name)]
                )
        return result_df


class StatisticalPreprocessorMixin:
    """Mixin providing statistical preprocessing utilities."""

    def normalize_columns(
        self, df: pl.DataFrame, columns: List[str], method: str = "standard"
    ) -> pl.DataFrame:
        """Normalize specified columns."""
        result_df = df
        for col in columns:
            if col not in df.columns:
                continue
            col_series = pl.col(col)
            if method == "standard":
                mean_val = df.select(col_series.mean()).item()
                std_val = df.select(col_series.std()).item()
                if std_val > 0:
                    result_df = result_df.with_columns(
                        [((col_series - mean_val) / std_val).alias(f"{col}_norm")]
                    )
        return result_df

    @staticmethod
    def outlier_removal_sigma(
        df: pl.DataFrame, column: str, n_sigma: float = 3.0
    ) -> pl.DataFrame:
        """Remove statistical outliers using sigma clipping."""
        col_series = pl.col(column)
        mean_val = df.select(col_series.mean()).item()
        std_val = df.select(col_series.std()).item()
        lower_bound = mean_val - n_sigma * std_val
        upper_bound = mean_val + n_sigma * std_val
        return df.filter(col_series.is_between(lower_bound, upper_bound))
