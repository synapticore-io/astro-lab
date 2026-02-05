"""
Test configuration and fixtures for AstroLab.

Uses real data from the data module for comprehensive testing.
"""

"""
Test configuration and fixtures for AstroLab.

Uses real data from the data module for comprehensive testing.
"""

from pathlib import Path

import polars as pl
import pytest

from astro_lab.config import get_data_config
from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from astro_lab.data.dataset.lightning import AstroLabDataModule
from astro_lab.data.info import SurveyInfo
from astro_lab.data.preprocessors.gaia import GaiaPreprocessor
from astro_lab.models import AstroModel

# Get central data configuration
data_config = get_data_config()


@pytest.fixture(scope="session")
def data_dir():
    """Test data directory."""
    return Path(data_config["base_dir"])


@pytest.fixture(scope="session")
def processed_dir():
    """Processed data directory."""
    return Path(data_config["processed_dir"])


@pytest.fixture(scope="session")
def survey_configs():
    """Available survey configurations."""
    return SurveyInfo().list_available_surveys()


@pytest.fixture(scope="session")
def small_survey():
    """Use GAIA as default survey for all tests."""
    return "gaia"


@pytest.fixture
def astro_datamodule(small_survey):
    """Create AstroLabDataModule with small dataset."""
    # Create dataset
    dataset = AstroLabInMemoryDataset(
        survey_name=small_survey,
    )

    # Create datamodule
    return AstroLabDataModule(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )


@pytest.fixture
def survey_graph_dataset(small_survey, processed_dir):
    """Create AstroLabInMemoryDataset with real data."""
    return AstroLabInMemoryDataset(
        survey_name=small_survey,
    )


@pytest.fixture
def real_survey_data(small_survey, processed_dir):
    """Load real survey data as Polars DataFrame from harmonized parquet file."""
    import os

    parquet_path = os.path.join(processed_dir, small_survey, f"{small_survey}.parquet")
    if not os.path.exists(parquet_path):
        pytest.skip(f"No harmonized parquet file found for survey: {small_survey}")
    return pl.read_parquet(parquet_path)


@pytest.fixture
def preprocessor(small_survey):
    """Get preprocessor for the default survey."""
    if small_survey == "gaia":
        return GaiaPreprocessor()
    else:
        # For other surveys, use Gaia as default for now
        return GaiaPreprocessor()


@pytest.fixture
def lightning_model():
    """Create a simple Lightning model for testing."""
    # Create a simple AstroModel for testing
    model = AstroModel(
        num_features=10,
        num_classes=3,
        hidden_dim=32,
        num_layers=2,
        task="node_classification",
    )
    return model


@pytest.fixture(scope="session")
def available_surveys():
    """Check which surveys have data available."""
    surveys = []
    for survey in SurveyInfo().list_available_surveys():
        # Check if processed data exists using central config
        processed_path = (
            Path(data_config["processed_dir"]) / survey / f"{survey}.parquet"
        )
        if processed_path.exists():
            surveys.append(survey)
    return surveys


@pytest.fixture
def real_graph_data(survey_graph_dataset):
    """Get real graph data from dataset."""
    if len(survey_graph_dataset) > 0:
        return survey_graph_dataset[0]
    else:
        pytest.skip("No real graph data available")
