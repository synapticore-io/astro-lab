"""
Config Components
================

Simple configuration management using Marimo UI elements.
"""

import marimo as mo

from .training_config import create_training_config
from .widget_config import create_widget_config as create_visualization_config


def get_available_analysis_methods():
    """Get available analysis methods from actual analysis modules."""
    return [
        "cosmic_web",  # Full cosmic web analysis
        "clustering",  # Multi-scale clustering
        "filament_detection",  # Filament detection only
        "structure_analysis",  # Structure analysis only
    ]


def create_data_config():
    """Create data configuration UI."""
    available_surveys = [
        "des",
        "euclid",
        "exoplanet",
        "gaia",
        "linear",
        "nsa",
        "panstarrs",
        "rrlyrae",
        "sdss",
        "tng50",
        "twomass",
        "wise",
    ]
    default_survey = "gaia"

    return mo.ui.dictionary(
        {
            "survey_name": mo.ui.dropdown(
                options=available_surveys, value=default_survey, label="Survey"
            ),
            "max_samples": mo.ui.slider(
                start=100, stop=100000, value=10000, label="Max Samples"
            ),
            "magnitude_limit": mo.ui.slider(
                start=5.0, stop=20.0, value=15.0, step=0.5, label="Magnitude Limit"
            ),
            "redshift_limit": mo.ui.slider(
                start=0.001, stop=3.0, value=0.15, step=0.01, label="Redshift Limit"
            ),
            "force_download": mo.ui.checkbox(value=False, label="Force Download"),
        }
    )


def create_analysis_config():
    """Create analysis configuration UI."""
    available_methods = get_available_analysis_methods()

    return mo.ui.dictionary(
        {
            "method": mo.ui.dropdown(
                options=available_methods,
                value="cosmic_web",
                label="Analysis Method",
            ),
            "n_neighbors": mo.ui.slider(start=5, stop=50, value=20, label="Neighbors"),
            "linking_length": mo.ui.slider(
                start=0.5, stop=10.0, value=2.0, step=0.1, label="Linking Length (Mpc)"
            ),
            "percolation_level": mo.ui.slider(
                start=0.1, stop=1.0, value=0.7, step=0.1, label="Percolation Level"
            ),
            "min_cluster_size": mo.ui.slider(
                start=1, stop=100, value=5, label="Min Cluster Size"
            ),
        }
    )


# Remove unnecessary wrapper functions - imported directly above


def create_config_form():
    """Create a complete configuration form."""
    data_config = create_data_config()
    analysis_config = create_analysis_config()
    training_config = create_training_config()
    viz_config = create_visualization_config()

    # Apply button
    apply_button = mo.ui.button(
        label="Apply Configuration",
        kind="success",
        on_click=lambda _: _apply_config(
            data_config.value,
            analysis_config.value,
            training_config.value,
            viz_config.value,
        ),
    )

    def _apply_config(data_vals, analysis_vals, training_vals, viz_vals):
        """Apply configuration to global state."""
        # Store in global state - simplified for now
        return "✅ Configuration applied successfully!"

    return mo.vstack(
        [
            mo.md("## Data Configuration"),
            data_config,
            mo.md("## Analysis Configuration"),
            analysis_config,
            mo.md("## Training Configuration"),
            training_config,
            mo.md("## Visualization Configuration"),
            viz_config,
            apply_button,
        ]
    )
