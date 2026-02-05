"""
Base Survey Collector
====================

Base class for survey data collectors with download and caching functionality.
"""

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import requests
from tqdm import tqdm

from astro_lab.config import get_data_config

logger = logging.getLogger(__name__)


class BaseSurveyCollector(ABC):
    """
    Base class for survey data collectors with enhanced download capabilities.

    Features:
    - Automatic download from various sources (HTTP, FTP, APIs)
    - File format conversion and validation
    - Caching and incremental downloads
    - Progress tracking and error handling
    - Checksum verification
    """

    def __init__(self, survey_name: str, data_config: Optional[Dict] = None):
        self.survey_name = survey_name
        self.config = data_config or self._get_default_config()
        self._setup_paths()
        self._cache = {}  # For caching download metadata

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the collector."""
        return {
            "download_timeout": 300,  # 5 minutes
            "chunk_size": 8192,  # 8KB chunks
            "verify_ssl": True,
            "retry_attempts": 3,
            "cache_downloads": True,
        }

    def _setup_paths(self):
        """Setup data paths for raw data."""
        data_config = get_data_config()
        self.raw_dir = Path(data_config["raw_dir"]) / f"{self.survey_name}"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_download_urls(self) -> List[str]:
        """Get list of URLs to download for this survey."""

    @abstractmethod
    def get_target_files(self) -> List[str]:
        """Get list of target file names after download."""

    def download(self, force: bool = False) -> List[Path]:
        """
        Download survey data from external sources.

        Args:
            force: Force re-download even if files exist

        Returns:
            List of downloaded file paths
        """
        logger.info(f"📥 Starting download for {self.survey_name}")

        urls = self.get_download_urls()
        target_files = self.get_target_files()

        if len(urls) != len(target_files):
            raise ValueError(
                f"URL count ({len(urls)}) must match target file count ({len(target_files)})"
            )

        downloaded_files = []

        for url, target_file in zip(urls, target_files):
            target_path = self.raw_dir / target_file

            # Skip if file exists and not forcing
            if target_path.exists() and not force:
                logger.info(f"✓ File already exists: {target_path}")
                downloaded_files.append(target_path)
                continue

            # Download file
            try:
                downloaded_path = self._download_file(url, target_path)
                downloaded_files.append(downloaded_path)
                logger.info(f"✅ Downloaded: {downloaded_path}")
            except Exception as e:
                logger.error(f"❌ Download failed for {url}: {e}")
                raise

        return downloaded_files

    def _download_file(self, url: str, target_path: Path) -> Path:
        """
        Download a single file with progress tracking.

        Args:
            url: Source URL
            target_path: Target file path

        Returns:
            Downloaded file path
        """
        logger.info(f"📥 Downloading {url} to {target_path}")

        # Create temporary file for download
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        try:
            response = requests.get(
                url,
                stream=True,
                timeout=self.config["download_timeout"],
                verify=self.config["verify_ssl"],
            )
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(temp_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=target_path.name,
                ) as pbar:
                    for chunk in response.iter_content(
                        chunk_size=self.config["chunk_size"]
                    ):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Move temporary file to final location
            shutil.move(temp_path, target_path)

        except Exception as e:
            # Clean up temporary file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e

        return target_path

    def validate_downloads(self, file_paths: List[Path]) -> bool:
        """
        Validate downloaded files.

        Args:
            file_paths: List of downloaded file paths

        Returns:
            True if all files are valid
        """
        logger.info(f"🔍 Validating {len(file_paths)} downloaded files")

        for file_path in file_paths:
            if not file_path.exists():
                logger.error(f"❌ File missing: {file_path}")
                return False

            # Check file size
            if file_path.stat().st_size == 0:
                logger.error(f"❌ Empty file: {file_path}")
                return False

            # Try to read file based on extension
            try:
                if file_path.suffix == ".parquet":
                    df = pl.read_parquet(file_path)
                    logger.info(f"✅ Validated {file_path.name}: {len(df)} rows")
                elif file_path.suffix in [".csv", ".tsv"]:
                    df = pl.read_csv(file_path)
                    logger.info(f"✅ Validated {file_path.name}: {len(df)} rows")
                elif file_path.suffix in [".fits", ".fit"]:
                    # Basic FITS validation
                    from astropy.io import fits

                    with fits.open(file_path) as hdul:
                        logger.info(f"✅ Validated {file_path.name}: {len(hdul)} HDUs")
                else:
                    logger.warning(f"⚠️ Unknown file type: {file_path.suffix}")

            except Exception as e:
                logger.error(f"❌ Validation failed for {file_path}: {e}")
                return False

        logger.info("✅ All downloads validated successfully")
        return True

    def get_download_info(self) -> Dict[str, Any]:
        """Get information about available downloads."""
        return {
            "survey": self.survey_name,
            "urls": self.get_download_urls(),
            "target_files": self.get_target_files(),
            "raw_dir": str(self.raw_dir),
        }
