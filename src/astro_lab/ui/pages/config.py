"""
Config Page
==========

Configuration management page.
"""

import marimo as mo

from astro_lab.ui.components import system_info


def create_page(app_state=None):
    """Create the configuration page."""

    # System status
    sys_status = system_info.create_system_status()

    # Configuration sections
    data_config = create_data_config(app_state)
    analysis_config = create_analysis_config(app_state)
    training_config = create_training_config(app_state)

    # Apply button
    apply_btn = mo.ui.button("💾 Apply Configuration", kind="success", full_width=True)

    status = mo.md("")

    if apply_btn.value and app_state:
        # Update app state with configurations
        app_state.data_config = {
            "survey_name": "gaia",  # From controls
            "max_samples": 10000,
        }
        app_state.analysis_config = {
            "method": "cosmic_web",
            "clustering_scales": [5.0, 10.0, 25.0],
        }
        app_state.training_config = {
            "model_type": "AstroGraphGNN",
            "epochs": 100,
        }

        status = mo.callout("✅ Configuration applied successfully!", kind="success")

    # Layout
    return mo.vstack(
        [
            mo.md("## ⚙️ Configuration"),
            mo.tabs(
                {
                    "System": sys_status,
                    "Data": data_config,
                    "Analysis": analysis_config,
                    "Training": training_config,
                }
            ),
            apply_btn,
            status,
        ]
    )


def create_data_config(app_state):
    """Create data configuration controls."""
    survey = mo.ui.dropdown(
        options=["gaia", "sdss", "nsa", "exoplanet"],
        value="gaia",
        label="Default Survey",
    )

    max_samples = mo.ui.number(
        100, 1000000, 10000, step=100, label="Default Sample Size"
    )

    return mo.vstack([mo.md("### 📊 Data Configuration"), survey, max_samples])


def create_analysis_config(app_state):
    """Create analysis configuration controls."""
    method = mo.ui.dropdown(
        options=["cosmic_web", "clustering", "filaments"],
        value="cosmic_web",
        label="Analysis Method",
    )

    min_samples = mo.ui.slider(3, 20, 5, label="Min Cluster Size")

    return mo.vstack([mo.md("### 🔬 Analysis Configuration"), method, min_samples])


def create_training_config(app_state):
    """Create training configuration controls."""
    model = mo.ui.dropdown(
        options=["AstroGraphGNN", "AstroNodeGNN", "AstroPointNet"],
        value="AstroGraphGNN",
        label="Model Architecture",
    )

    epochs = mo.ui.slider(10, 500, 100, step=10, label="Training Epochs")

    batch_size = mo.ui.dropdown(options=[16, 32, 64, 128], value=32, label="Batch Size")

    return mo.vstack(
        [mo.md("### 🤖 Training Configuration"), model, epochs, batch_size]
    )
