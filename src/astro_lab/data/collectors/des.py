"""
DES Survey Collector
===================

Collector for DES (Dark Energy Survey) data using astroquery sources.
Only real astronomical data - no synthetic generation.
"""

import logging
from pathlib import Path
from typing import List

import astropy.units as u
import polars as pl
from astropy.coordinates import SkyCoord

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class DESCollector(BaseSurveyCollector):
    """
    Collector for DES data using astronomical data sources.
    Downloads real galaxy data from SIMBAD within the DES footprint.
    """

    def __init__(self, survey_name: str = "des", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        """No direct download URLs - uses astroquery."""
        return []

    def get_target_files(self) -> List[str]:
        """Target files to be created."""
        return ["des_galaxies.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download real DES galaxy data from SIMBAD for cosmic web analysis.

        Args:
            force: Force re-download even if file exists

        Returns:
            List of downloaded file paths
        """
        logger.info("📥 Downloading DES galaxy data from SIMBAD...")
        target_parquet = self.raw_dir / "des_galaxies.parquet"

        if target_parquet.exists() and not force:
            logger.info(f"✓ DES data already exists: {target_parquet}")
            return [target_parquet]

        try:
            from astroquery.simbad import Simbad
        except ImportError:
            logger.error("astroquery is required for DES data collection")
            raise ImportError("Please install astroquery: pip install astroquery")

        # Configure SIMBAD for galaxy queries
        simbad = Simbad()

        # Add available votable fields for galaxies
        available_fields = set(simbad.get_votable_fields())
        logger.info(f"Available SIMBAD fields: {len(available_fields)}")

        # Only add fields that are actually available
        fields_to_add = []
        for field in ["otype", "z_value", "flux(V)", "flux(B)", "flux(R)", "flux(I)"]:
            if field in available_fields:
                fields_to_add.append(field)

        for field in fields_to_add:
            simbad.add_votable_fields(field)

        logger.info(f"Querying with fields: {fields_to_add}")

        # DES footprint regions: Southern hemisphere, specific RA/Dec ranges
        # Split into smaller regions to avoid timeouts
        des_regions = [
            (30.0, -30.0, 8.0),  # Eastern region
            (45.0, -25.0, 8.0),  # Central-east region
            (60.0, -20.0, 8.0),  # Central region
            (75.0, -25.0, 8.0),  # Central-west region
            (90.0, -30.0, 8.0),  # Western region
        ]

        all_results = []

        for ra_center, dec_center, radius in des_regions:
            try:
                coord = SkyCoord(
                    ra=ra_center, dec=dec_center, unit=(u.deg, u.deg), frame="icrs"
                )

                logger.info(
                    f"📡 Querying SIMBAD region: RA={ra_center}°, Dec={dec_center}°, "
                    f"Radius={radius}°"
                )

                # Query for objects in this region
                result = simbad.query_region(coord, radius=radius * u.deg)

                if result is None or len(result) == 0:
                    logger.warning(
                        f"No objects found in region RA={ra_center}, Dec={dec_center}"
                    )
                    continue

                # Filter for galaxies if object type information is available
                if "OTYPE" in result.colnames:
                    # Filter for galaxy types
                    galaxy_mask = []
                    for obj_type in result["OTYPE"]:
                        if obj_type is not None:
                            # Galaxy types in SIMBAD: G, GiC, GiG, etc.
                            is_galaxy = str(obj_type).upper().startswith("G")
                            galaxy_mask.append(is_galaxy)
                        else:
                            galaxy_mask.append(False)

                    filtered_result = result[galaxy_mask]
                    logger.info(
                        f"Filtered to {len(filtered_result)} galaxies from "
                        f"{len(result)} total objects"
                    )
                else:
                    # No object type info, use all results
                    filtered_result = result
                    logger.info(
                        f"No object type filtering, using all {len(result)} objects"
                    )

                # Further filter for objects with redshift if available
                if "Z_VALUE" in filtered_result.colnames:
                    redshift_mask = [
                        z is not None and not str(z).strip() == ""
                        for z in filtered_result["Z_VALUE"]
                    ]
                    redshift_filtered = filtered_result[redshift_mask]

                    if len(redshift_filtered) > 0:
                        filtered_result = redshift_filtered
                        logger.info(
                            f"Found {len(filtered_result)} objects with redshift data"
                        )

                if len(filtered_result) > 0:
                    # Convert to Polars DataFrame
                    df = pl.from_pandas(filtered_result.to_pandas())

                    # Add region identifier for tracking
                    df = df.with_columns(
                        pl.lit(f"des_region_{ra_center}_{dec_center}").alias(
                            "source_region"
                        )
                    )

                    all_results.append(df)
                    logger.info(
                        f"✅ Added {len(df)} objects from region "
                        f"RA={ra_center}, Dec={dec_center}"
                    )
                else:
                    logger.warning(
                        f"No suitable objects found in region "
                        f"RA={ra_center}, Dec={dec_center}"
                    )

            except Exception as e:
                logger.error(
                    f"❌ SIMBAD query failed for region RA={ra_center}, "
                    f"Dec={dec_center}: {e}"
                )
                continue

        if not all_results:
            logger.error(
                "No real data could be downloaded from any region. "
                "Please check network connection and try again."
            )
            raise RuntimeError(
                "Failed to download any real DES data from SIMBAD. "
                "This may be due to network issues or SIMBAD availability."
            )

        # Combine all results
        combined_df = pl.concat(all_results, how="vertical")

        # Remove duplicates if any
        if "MAIN_ID" in combined_df.columns:
            initial_count = len(combined_df)
            combined_df = combined_df.unique(subset=["MAIN_ID"])
            final_count = len(combined_df)
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} duplicate objects")

        logger.info(f"✅ Downloaded total {len(combined_df)} real objects from SIMBAD")

        # Save to parquet
        combined_df.write_parquet(target_parquet)
        logger.info(f"✅ DES data saved to: {target_parquet}")

        return [target_parquet]

    def _validate_downloaded_data(self, file_path: Path) -> bool:
        """Validate that downloaded data is real and usable."""
        try:
            df = pl.read_parquet(file_path)

            # Check basic requirements
            if len(df) == 0:
                logger.error("Downloaded file is empty")
                return False

            # Check for coordinate columns
            required_coords = ["RA", "DEC"]  # SIMBAD standard column names
            missing_coords = [col for col in required_coords if col not in df.columns]

            if missing_coords:
                logger.error(f"Missing coordinate columns: {missing_coords}")
                return False

            # Check coordinate ranges are reasonable
            ra_values = df["RA"].to_numpy()
            dec_values = df["DEC"].to_numpy()

            if not (0 <= ra_values.min() and ra_values.max() <= 360):
                logger.error(
                    f"Invalid RA range: {ra_values.min()} to {ra_values.max()}"
                )
                return False

            if not (-90 <= dec_values.min() and dec_values.max() <= 90):
                logger.error(
                    f"Invalid Dec range: {dec_values.min()} to {dec_values.max()}"
                )
                return False

            logger.info(f"✅ Data validation passed: {len(df)} real objects")
            logger.info(f"RA range: {ra_values.min():.2f} to {ra_values.max():.2f}")
            logger.info(f"Dec range: {dec_values.min():.2f} to {dec_values.max():.2f}")

            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
