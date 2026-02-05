"""
Training Component
=================

Training interface for AstroLab models.
"""

from pathlib import Path

import marimo as mo

from astro_lab.config import get_data_paths
from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer
from astro_lab.ui.components import state

from .config import create_training_config


def create_training_form():
    """Create training form."""
    # Create configuration UI
    config_ui = create_training_config()

    # Create train button
    train_button = mo.ui.button(
        label="🏋️ Start Training",
        kind="success",
        on_click=lambda _: _start_training(config_ui.value),
    )

    def _start_training(config):
        """Start training using AstroLab training modules."""
        try:
            # Get loaded data
            current_state = state.get_state()
            loaded_data = current_state.get("loaded_data")
            if loaded_data is None or loaded_data.is_empty():
                return "❌ No data loaded. Please load data first."

            datamodule = loaded_data

            # Create trainer
            trainer = AstroTrainer(
                experiment_name=config["experiment_name"],
                run_name=config.get("run_name"),
                survey=datamodule.survey,
                task=config.get("task", "node_classification"),
                checkpoint_dir=str(
                    Path(get_data_paths()["checkpoint_dir"]) / config["experiment_name"]
                ),
                max_epochs=config["epochs"],
                devices=config.get("devices", 1),
                accelerator=config.get("accelerator", "auto"),
            )

            # Create model based on configuration
            model = _create_model_from_config(config, datamodule)

            # Start training
            trainer.fit(model, datamodule)

            # Update state
            state.set_state(
                {
                    "training_result": trainer,
                    "trained_model": model,
                    "status": (
                        f"✅ Completed training for {config['model_type']} model"
                    ),
                }
            )
            return (
                f"✅ Successfully completed training for {config['model_type']} model!"
            )

        except Exception as e:
            return f"❌ Error in training: {str(e)}"

    return mo.vstack([config_ui, train_button])


def _create_model_from_config(config, datamodule):
    """Create model from configuration and datamodule."""
    # Get model parameters from datamodule
    num_features = datamodule.num_features
    num_classes = datamodule.num_classes

    # Model configuration
    model_kwargs = {
        "num_features": num_features,
        "num_classes": num_classes,
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "dropout": config.get("dropout", 0.1),
        "task": config.get("task", "node_classification"),
        "conv_type": config.get("conv_type", "gat"),
        "heads": config.get("heads", 4),
        "pooling": config.get("pooling", "mean"),
        "activation": config.get("activation", "gelu"),
        "norm": config.get("norm", "layer"),
        "residual": config.get("residual", True),
    }

    # Create model based on type
    model_type = config.get("model_type", "stellar")

    if model_type == "stellar":
        model = AstroModel(**model_kwargs)
    elif model_type == "cosmic_web":
        from astro_lab.models import create_cosmic_web_model

        model = create_cosmic_web_model(**model_kwargs)
    elif model_type == "galaxy":
        from astro_lab.models import create_galaxy_model

        model = create_galaxy_model(**model_kwargs)
    elif model_type == "exoplanet":
        from astro_lab.models import create_exoplanet_model

        model = create_exoplanet_model(**model_kwargs)
    else:
        # Default to generic AstroModel
        model = AstroModel(**model_kwargs)

    return model


def create_training_status():
    """Create training status display."""
    current_state = state.get_state()

    if current_state.get("training_result"):
        status_text = "**Training Status:** ✅ Completed\n"
        status_text += (
            f"**Model Type:** {type(current_state['training_result']).__name__}"
        )
    else:
        status_text = "**Training Status:** ⏳ No training run yet"

    return mo.md(status_text)
