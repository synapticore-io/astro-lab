"""
UI State Management for AstroLab
===============================

Modern state management using Marimo's reactive patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppState:
    """Application state container with reactive updates."""

    # Data state
    loaded_data: Optional[Any] = None
    selected_survey: str = "gaia"
    data_format: str = "dataframe"
    n_objects: int = 0
    status: str = "Ready"

    # Analysis state
    analysis_result: Optional[Dict[str, Any]] = None
    analysis_status: str = "Not started"
    analysis_method: str = "cosmic_web"

    # Visualization state
    visualization_result: Optional[Any] = None
    viz_backend: str = "cosmograph"

    # Training state
    training_result: Optional[Dict[str, Any]] = None
    training_status: str = "Not started"
    current_epoch: int = 0
    total_epochs: int = 100
    current_loss: float = 0.0
    current_accuracy: float = 0.0

    # Config state
    data_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "survey_name": "gaia",
            "max_samples": 10000,
            "magnitude_limit": 15.0,
        }
    )

    analysis_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "method": "cosmic_web",
            "clustering_scales": [5.0, 10.0, 25.0, 50.0],
            "min_samples": 5,
        }
    )

    training_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model_type": "AstroGraphGNN",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
    )

    visualization_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "cosmograph",
            "node_size": 2.0,
            "edge_opacity": 0.3,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            # Data
            "loaded_data": f"{type(self.loaded_data).__name__}"
            if self.loaded_data
            else None,
            "selected_survey": self.selected_survey,
            "data_format": self.data_format,
            "n_objects": self.n_objects,
            "status": self.status,
            # Analysis
            "analysis_status": self.analysis_status,
            "analysis_method": self.analysis_method,
            "has_analysis_result": self.analysis_result is not None,
            # Visualization
            "viz_backend": self.viz_backend,
            "has_viz_result": self.visualization_result is not None,
            # Training
            "training_status": self.training_status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": self.current_loss,
            "current_accuracy": self.current_accuracy,
            # Configs
            "data_config": self.data_config,
            "analysis_config": self.analysis_config,
            "training_config": self.training_config,
            "visualization_config": self.visualization_config,
        }


# Create global state using Marimo's state management
def create_state() -> AppState:
    """Create reactive application state."""
    return AppState()


# Helper functions for backward compatibility
def get_state() -> Dict[str, Any]:
    """Get current state as dictionary (for backward compatibility)."""
    # This should be replaced with proper state passing
    return {
        "loaded_data": None,
        "selected_survey": "gaia",
        "status": "Ready",
        "analysis_result": None,
        "visualization_result": None,
        "training_result": None,
        "n_objects": 0,
        "current_survey": "None",
    }


def set_state(new_state: Dict[str, Any]):
    """Set state (for backward compatibility)."""
    # This should be replaced with proper state updates


def update_state(**kwargs):
    """Update state fields (for backward compatibility)."""
    # This should be replaced with proper state updates


# Config helpers
def get_data_config() -> Dict[str, Any]:
    """Get default data configuration."""
    return {
        "survey_name": "gaia",
        "max_samples": 10000,
        "magnitude_limit": 15.0,
        "redshift_limit": 0.15,
        "force_download": False,
    }


def get_analysis_config() -> Dict[str, Any]:
    """Get default analysis configuration."""
    return {
        "method": "cosmic_web",
        "clustering_algorithm": "dbscan",
        "clustering_scales": [5.0, 10.0, 25.0, 50.0],
        "min_samples": 5,
        "filament_method": "mst",
        "detect_voids": True,
    }


def get_training_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        "model_type": "AstroGraphGNN",
        "task": "node_classification",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "scheduler": "cosine_annealing",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "gradient_clip_val": 1.0,
        "precision": "16-mixed",
        "accelerator": "auto",
    }


def get_visualization_config() -> Dict[str, Any]:
    """Get default visualization configuration."""
    return {
        "backend": "cosmograph",
        "node_size": 2.0,
        "node_color": "auto",
        "edge_opacity": 0.3,
        "show_edges": True,
        "layout": "force",
        "dimension": "3d",
    }
