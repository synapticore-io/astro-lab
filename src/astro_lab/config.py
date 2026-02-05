"""
Central Configuration for AstroLab
=================================

Single source of truth for all AstroLab configuration.
Config YAMLs are loaded from the top-level 'configs' directory.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# Project root logic (for rare use, e.g. find_project_root())


def find_project_root():
    """Find the project root directory (where pyproject.toml is located)."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent


# Config YAML loading


def load_yaml(filename):
    """Load YAML config file with proper error handling."""
    path = Path(__file__).parent.parent.parent / "configs" / filename
    if not path.exists():
        return None
    with open(path) as f:
        content = yaml.safe_load(f)
        # Handle empty YAML files
        return content if content else {}


SURVEY_CONFIGS = {}
for section in [
    "data",
    "model",
    "training",
    "mlflow",
    "tasks",
    "hpo",
    "albpy",
    "surveys",
]:
    loaded = load_yaml(f"{section}.yaml")
    if loaded is not None:
        if section == "surveys":
            SURVEY_CONFIGS[section] = loaded
        else:
            # Handle both nested and flat config structures
            if isinstance(loaded, dict) and section in loaded:
                SURVEY_CONFIGS[section] = loaded[section]
            else:
                # For flat configs or empty configs
                SURVEY_CONFIGS[section] = loaded

# Config getter functions


def get_config() -> Dict[str, Any]:
    """Get the merged configuration from all YAMLs."""
    return SURVEY_CONFIGS


def get_data_config() -> dict:
    return SURVEY_CONFIGS.get("data", {})


def get_model_config() -> dict:
    return SURVEY_CONFIGS.get("model", {})


def get_training_config() -> dict:
    return SURVEY_CONFIGS.get("training", {})


def get_mlflow_config() -> dict:
    return SURVEY_CONFIGS.get("mlflow", {})


def get_task_config(task: str) -> dict:
    tasks = SURVEY_CONFIGS.get("tasks", {})
    if task not in tasks:
        # Return sensible defaults for unknown tasks
        return {
            "task": task,
            "conv_type": "gcn",
            "pooling": "mean" if "graph" in task else None,
        }
    return tasks[task]


def get_hpo_config() -> dict:
    return SURVEY_CONFIGS.get("hpo", {})


def get_albpy_config() -> dict:
    return SURVEY_CONFIGS.get("albpy", {})


def get_survey_config(survey: str) -> dict:
    surveys = SURVEY_CONFIGS.get("surveys", {})
    if survey not in surveys:
        raise ValueError(f"Survey '{survey}' not found in surveys config.")
    return surveys[survey]


def get_combined_config(survey: str, task: str) -> Dict[str, Any]:
    """Get combined configuration for survey and task."""
    # Start with base configs
    data_config = get_data_config()
    model_config = get_model_config()
    training_config = get_training_config()
    mlflow_config = get_mlflow_config()

    # Get survey-specific config
    survey_config = get_survey_config(survey)

    # Get task-specific config
    task_config = get_task_config(task)

    # Merge configs with priority: Survey > Task > General
    combined = {}

    # Base configs
    combined.update(data_config)
    combined.update(model_config)
    combined.update(training_config)
    combined.update(mlflow_config)

    # Task-specific overrides
    combined.update(task_config)

    # Survey-specific overrides (highest priority for survey fields)
    # But don't override all fields, just the ones that make sense
    survey_overrides = {
        "batch_size": survey_config.get("batch_size"),
        "k_neighbors": survey_config.get("k_neighbors"),
        "precision": survey_config.get("precision"),
        "gradient_clip_val": survey_config.get("gradient_clip_val"),
        "experiment_name": survey_config.get("experiment_name"),
    }

    # Add non-None survey overrides
    for key, value in survey_overrides.items():
        if value is not None:
            combined[key] = value

    # If survey has recommended model and no model type specified yet, use it
    if "recommended_model" in survey_config and "conv_type" not in combined:
        combined.update(survey_config["recommended_model"])

    # Ensure task is set
    combined["task"] = task

    return combined


def get_checkpoint_filename(
    survey: str,
    task: str,
    epoch: int,
    metric_value: float,
    metric_name: str = "val_loss",
) -> str:
    """Generate checkpoint filename."""
    return f"{survey}_{task}_epoch{epoch}_{metric_name}{metric_value:.4f}"


def get_model_name(survey: str, task: str) -> str:
    """Generate model name."""
    return f"{survey}_{task}_model"


def get_run_name(survey: str, task: str, model_type: str = "astro_model") -> str:
    """Generate run name."""
    return f"{survey}_{task}_{model_type}"


def get_optimal_batch_size(gpu_memory_gb: Optional[float] = None) -> int:
    """Get optimal batch size based on GPU memory."""
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb:
        if gpu_memory_gb < 8:
            return 16
        elif gpu_memory_gb < 16:
            return 32
        else:
            return 64
    return 32  # Default


def get_data_paths() -> Dict[str, str]:
    """Get all relevant data paths from the YAML config."""
    data_cfg = get_data_config()
    mlflow_cfg = get_mlflow_config()
    return {
        "base_dir": data_cfg.get("base_dir", "data"),
        "raw_dir": data_cfg.get("raw_dir", "data/raw"),
        "processed_dir": data_cfg.get("processed_dir", "data/processed"),
        "cache_dir": data_cfg.get("cache_dir", "data/cache"),
        "checkpoint_dir": data_cfg.get("checkpoint_dir", "data/checkpoints"),
        "mlruns_dir": mlflow_cfg.get("tracking_uri", "data/mlruns"),
    }
