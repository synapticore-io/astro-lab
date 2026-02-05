"""
Training Configuration
=====================

Configuration for all available training options, optimizers, and schedulers.
"""

import marimo as mo

from astro_lab.config import get_config, get_training_config


def create_training_config():
    """Create training configuration UI."""
    # Build tasks from config
    config = get_config()
    tasks_dict = config.get("tasks", {})
    tasks = list(tasks_dict.keys())
    # Presets: use model config keys or hardcode a default
    model_config = config.get("model", {})
    presets = (
        list(model_config.keys()) if isinstance(model_config, dict) else ["default"]
    )

    return mo.ui.dictionary(
        {
            "preset": mo.ui.dropdown(
                options=presets,
                value=presets[0] if presets else "default",
                label="Model Preset",
            ),
            "task": mo.ui.dropdown(
                options=tasks,
                value=tasks[0] if tasks else "node_classification",
                label="Task",
            ),
            "epochs": mo.ui.slider(
                start=1,
                stop=500,
                value=get_training_config().get("max_epochs", 100),
                label="Epochs",
            ),
            "batch_size": mo.ui.slider(
                start=8,
                stop=256,
                value=get_training_config().get("batch_size", 32),
                label="Batch Size",
            ),
            "learning_rate": mo.ui.slider(
                start=1e-5,
                stop=1e-1,
                value=get_training_config().get("learning_rate", 0.001),
                step=1e-4,
                label="Learning Rate",
            ),
            "optimizer": mo.ui.dropdown(
                options=["adamw", "adam", "sgd"],
                value=get_training_config().get("optimizer", "adamw"),
                label="Optimizer",
            ),
            "scheduler": mo.ui.dropdown(
                options=["onecycle", "cosine_annealing", "warmup_cosine"],
                value="cosine_annealing",
                label="Learning Rate Scheduler",
            ),
            "hidden_dim": mo.ui.slider(
                start=16,
                stop=512,
                value=get_training_config().get("hidden_dim", 128),
                label="Hidden Dimension",
            ),
            "num_layers": mo.ui.slider(
                start=1,
                stop=10,
                value=get_training_config().get("num_layers", 3),
                label="Number of Layers",
            ),
            "dropout": mo.ui.slider(
                start=0.0,
                stop=0.5,
                value=get_training_config().get("dropout", 0.1),
                step=0.05,
                label="Dropout",
            ),
            "weight_decay": mo.ui.slider(
                start=0.0,
                stop=1e-2,
                value=get_training_config().get("weight_decay", 0.01),
                step=1e-6,
                label="Weight Decay",
            ),
            "gradient_clip_val": mo.ui.slider(
                start=0.1,
                stop=10.0,
                value=get_training_config().get("gradient_clip_val", 1.0),
                step=0.1,
                label="Gradient Clipping",
            ),
            "precision": mo.ui.dropdown(
                options=["32", "16-mixed", "16"],
                value=get_training_config().get("precision", "16-mixed"),
                label="Precision",
            ),
            "accelerator": mo.ui.dropdown(
                options=["cpu", "gpu", "auto"],
                value=get_training_config().get("accelerator", "auto"),
                label="Accelerator",
            ),
        }
    )
