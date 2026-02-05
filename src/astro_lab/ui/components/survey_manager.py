"""
Survey Manager Component - Clean and reactive survey management
Following Marimo 2025 best practices with reactive state management
"""

from typing import Any, Dict, List, Optional

import marimo as mo
import numpy as np
import polars as pl

# Direct imports - no antipatterns
from astro_lab.data.collectors import (
    ExoplanetCollector,
    GaiaCollector,
    LINEARCollector,
    NSACollector,
    RRLyraeCollector,
    SDSSCollector,
    TNG50Collector,
    WISECollector,
)
from astro_lab.data.preprocessors import (
    ExoplanetPreprocessor,
    GaiaPreprocessor,
    LINEARPreprocessor,
    NSAPreprocessor,
    RRLyraePreprocessor,
    SDSSPreprocessor,
    TNG50Preprocessor,
    WISEPreprocessor,
)

# Survey Configuration - Clean definition
SURVEYS = {
    "gaia": {
        "name": "Gaia DR3",
        "icon": "⭐",
        "collector": GaiaCollector,
        "preprocessor": GaiaPreprocessor,
        "coord_type": "equatorial",
    },
    "sdss": {
        "name": "SDSS DR17",
        "icon": "🌌",
        "collector": SDSSCollector,
        "preprocessor": SDSSPreprocessor,
        "coord_type": "equatorial",
    },
    "nsa": {
        "name": "NASA-Sloan Atlas",
        "icon": "🌀",
        "collector": NSACollector,
        "preprocessor": NSAPreprocessor,
        "coord_type": "equatorial",
    },
    "exoplanet": {
        "name": "Exoplanet Archive",
        "icon": "🪐",
        "collector": ExoplanetCollector,
        "preprocessor": ExoplanetPreprocessor,
        "coord_type": "equatorial",
    },
    "tng50": {
        "name": "TNG50 Simulation",
        "icon": "🔬",
        "collector": TNG50Collector,
        "preprocessor": TNG50Preprocessor,
        "coord_type": "cartesian",
    },
    "wise": {
        "name": "WISE All-Sky",
        "icon": "🌡️",
        "collector": WISECollector,
        "preprocessor": WISEPreprocessor,
        "coord_type": "equatorial",
    },
    "linear": {
        "name": "LINEAR Asteroids",
        "icon": "☄️",
        "collector": LINEARCollector,
        "preprocessor": LINEARPreprocessor,
        "coord_type": "orbital",
    },
    "rrlyrae": {
        "name": "RR Lyrae Variables",
        "icon": "💫",
        "collector": RRLyraeCollector,
        "preprocessor": RRLyraePreprocessor,
        "coord_type": "equatorial",
    },
}


def create_survey_selector():
    """Create reactive survey selector component"""
    options = {f"{info['icon']} {info['name']}": key for key, info in SURVEYS.items()}

    return mo.ui.dropdown(options=options, label="Select Survey", value="gaia")


def create_sample_size_slider():
    """Create sample size control"""
    return mo.ui.slider(
        start=100, stop=10000, value=1000, step=100, label="Max Samples"
    )


def create_data_overview():
    """Create survey overview table"""
    survey_list = []

    for key, info in SURVEYS.items():
        survey_list.append(
            {
                "Survey": f"{info['icon']} {info['name']}",
                "Key": key.upper(),
                "Coordinates": info["coord_type"].title(),
                "Status": "✅ Available",
            }
        )

    df = pl.DataFrame(survey_list)

    return mo.ui.table(df.to_pandas(), selection=None, page_size=8)


def load_survey_data(
    survey_key: str, max_samples: int = 1000
) -> Optional[pl.DataFrame]:
    """
    Load survey data synchronously
    Returns None on error for clean error handling
    """
    try:
        survey_info = SURVEYS.get(survey_key)
        if not survey_info:
            raise ValueError(f"Survey {survey_key} not available")

        # Initialize collector and preprocessor
        collector = survey_info["collector"]()
        preprocessor = survey_info["preprocessor"]()

        # Collect and process data
        raw_data = collector.collect(max_samples=max_samples)
        processed_data = preprocessor.preprocess(raw_data)

        # Ensure Polars DataFrame
        if not isinstance(processed_data, pl.DataFrame):
            processed_data = pl.DataFrame(processed_data)

        return processed_data

    except Exception:
        # Clean error handling - return None
        return None


def check_spatial_data(data: pl.DataFrame) -> bool:
    """Check if DataFrame contains spatial coordinates"""
    spatial_cols = ["ra", "dec", "x", "y", "z", "coordinates"]
    return any(col in data.columns for col in spatial_cols)


def extract_coordinates(data: pl.DataFrame) -> Optional[np.ndarray]:
    """
    Extract spatial coordinates from DataFrame
    Returns None if no valid coordinates found
    """
    # RA/DEC coordinates (equatorial)
    if "ra" in data.columns and "dec" in data.columns:
        return np.column_stack(
            [
                data["ra"].to_numpy(),
                data["dec"].to_numpy(),
                np.zeros(len(data)),  # Z = 0 for sky coordinates
            ]
        )

    # Cartesian coordinates (XYZ)
    if "x" in data.columns and "y" in data.columns:
        z_col = data["z"].to_numpy() if "z" in data.columns else np.zeros(len(data))
        return np.column_stack([data["x"].to_numpy(), data["y"].to_numpy(), z_col])

    # Fallback: use first numeric columns
    numeric_cols = [
        col
        for col in data.columns
        if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ][:3]

    if len(numeric_cols) < 2:
        return None

    coord_data = [data[col].to_numpy() for col in numeric_cols]

    # Pad to 3D if needed
    while len(coord_data) < 3:
        coord_data.append(np.zeros(len(data)))

    return np.column_stack(coord_data)


def get_survey_info(survey_key: str) -> Optional[Dict[str, Any]]:
    """Get survey configuration info"""
    return SURVEYS.get(survey_key)


def get_available_survey_keys() -> List[str]:
    """Get list of available survey keys"""
    return list(SURVEYS.keys())


def create_data_loading_status(
    data: Optional[pl.DataFrame], survey_key: str
) -> mo.Html:
    """Create status display for loaded data"""
    if data is None:
        return mo.md("📭 **No data loaded**")

    survey_info = SURVEYS.get(survey_key, {})
    has_coords = check_spatial_data(data)

    status_md = f"""
    ## ✅ Data Loaded Successfully
    
    **Survey:** {survey_info.get("icon", "📊")} {survey_info.get("name", survey_key.upper())}
    **Objects:** {len(data):,}
    **Features:** {len(data.columns)}
    **Spatial Data:** {"✅ Available" if has_coords else "⚠️ None"}
    **Coordinate Type:** {survey_info.get("coord_type", "Unknown").title()}
    """

    return mo.md(status_md)
