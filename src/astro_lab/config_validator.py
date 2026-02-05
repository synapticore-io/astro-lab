"""Validate configuration consistency across all YAML files."""

from pathlib import Path
from typing import List, Tuple

from astro_lab.config import get_config


class ConfigValidator:
    """Validate AstroLab configuration files."""

    def __init__(self):
        self.configs = get_config()
        self.issues = []

    def validate_all(self) -> Tuple[bool, List[str]]:
        """Run all validation checks."""
        self.issues = []

        # Check individual configs
        self._validate_data_config()
        self._validate_model_config()
        self._validate_training_config()
        self._validate_survey_configs()
        self._validate_task_configs()

        # Check cross-config consistency
        self._validate_cross_config_consistency()

        return len(self.issues) == 0, self.issues

    def _validate_data_config(self):
        """Validate data configuration."""
        data = self.configs.get("data", {})

        # Check required fields
        required = ["base_dir", "processed_dir", "raw_dir"]
        for field in required:
            if field not in data:
                self.issues.append(f"data.yaml: Missing required field '{field}'")

        # Check paths exist
        for field in ["base_dir", "processed_dir", "raw_dir"]:
            if field in data:
                path = Path(data[field])
                # Only check base_dir existence, others are created on demand
                if field == "base_dir" and not path.exists():
                    self.issues.append(
                        f"data.yaml: Base directory '{data[field]}' does not exist"
                    )

        # Validate numeric ranges
        if data.get("batch_size", 0) < 1:
            self.issues.append("data.yaml: batch_size must be >= 1")

        if data.get("k_neighbors", 0) < 1:
            self.issues.append("data.yaml: k_neighbors must be >= 1")

    def _validate_model_config(self):
        """Validate model configuration."""
        model = self.configs.get("model", {})

        # Check conv_type is valid
        valid_conv_types = [
            "gcn",
            "gat",
            "gin",
            "sage",
            "transformer",
            "pointnet",
            "temporal",
        ]
        if "conv_type" in model and model["conv_type"] not in valid_conv_types:
            self.issues.append(
                f"model.yaml: conv_type '{model.get('conv_type')}' "
                f"not in {valid_conv_types}"
            )

        # Check numeric ranges
        if model.get("hidden_dim", 0) < 1:
            self.issues.append("model.yaml: hidden_dim must be >= 1")

        if model.get("num_layers", 0) < 1:
            self.issues.append("model.yaml: num_layers must be >= 1")

        if "dropout" in model and not 0 <= model["dropout"] <= 1:
            self.issues.append("model.yaml: dropout must be in [0, 1]")

    def _validate_training_config(self):
        """Validate training configuration."""
        training = self.configs.get("training", {})

        # Check numeric ranges
        if training.get("max_epochs", 0) < 1:
            self.issues.append("training.yaml: max_epochs must be >= 1")

        if training.get("learning_rate", 0) <= 0:
            self.issues.append("training.yaml: learning_rate must be > 0")

        # Validate precision values
        valid_precisions = ["32-true", "16-mixed", "bf16-mixed"]
        if "precision" in training and training["precision"] not in valid_precisions:
            self.issues.append(
                f"training.yaml: precision '{training['precision']}' "
                f"not in {valid_precisions}"
            )

    def _validate_survey_configs(self):
        """Validate each survey configuration."""
        surveys = self.configs.get("surveys", {})

        for survey_name, survey_config in surveys.items():
            # Check required fields
            required = ["name", "coord_cols", "mag_cols"]
            for field in required:
                if field not in survey_config:
                    self.issues.append(
                        f"surveys.yaml: Survey '{survey_name}' "
                        f"missing required field '{field}'"
                    )

            # Validate recommended model if present
            if "recommended_model" in survey_config:
                rec_model = survey_config["recommended_model"]

                if "conv_type" in rec_model:
                    valid_types = [
                        "gcn",
                        "gat",
                        "gin",
                        "sage",
                        "transformer",
                        "pointnet",
                        "temporal",
                    ]
                    if rec_model["conv_type"] not in valid_types:
                        self.issues.append(
                            f"surveys.yaml: Survey '{survey_name}' has invalid "
                            f"recommended conv_type '{rec_model['conv_type']}'"
                        )

                # Check GAT/Transformer have heads
                if rec_model.get("conv_type") in ["gat", "transformer"]:
                    if "heads" not in rec_model:
                        self.issues.append(
                            f"surveys.yaml: Survey '{survey_name}' with "
                            f"{rec_model['conv_type']} should specify 'heads'"
                        )

    def _validate_task_configs(self):
        """Validate task configurations."""
        tasks = self.configs.get("tasks", {})

        valid_tasks = [
            "node_classification",
            "graph_classification",
            "node_regression",
            "graph_regression",
            "link_prediction",
        ]

        for task_name, task_config in tasks.items():
            if task_name not in valid_tasks:
                self.issues.append(f"tasks.yaml: Unknown task '{task_name}'")

            # Check task consistency
            if task_config.get("task") != task_name:
                self.issues.append(
                    f"tasks.yaml: Task '{task_name}' has mismatched "
                    f"task field '{task_config.get('task')}'"
                )

            # Graph tasks should have pooling
            if "graph" in task_name and not task_config.get("pooling"):
                self.issues.append(
                    f"tasks.yaml: Graph task '{task_name}' should specify pooling"
                )

    def _validate_cross_config_consistency(self):
        """Check consistency across config files."""
        # Check that task conv_types are valid
        tasks = self.configs.get("tasks", {})
        self.configs.get("model", {})

        for task_name, task_config in tasks.items():
            if "conv_type" in task_config:
                # Should be a valid conv type
                valid_types = [
                    "gcn",
                    "gat",
                    "gin",
                    "sage",
                    "transformer",
                    "pointnet",
                    "temporal",
                ]
                if task_config["conv_type"] not in valid_types:
                    self.issues.append(
                        f"Cross-config: Task '{task_name}' has invalid "
                        f"conv_type '{task_config['conv_type']}'"
                    )

        # Check HPO search space references valid config fields
        hpo = self.configs.get("hpo", {})
        if "search_space" in hpo:
            for param, param_config in hpo["search_space"].items():
                # Check if parameter exists in model or training config
                if param not in [
                    "learning_rate",
                    "batch_size",
                    "hidden_dim",
                    "num_layers",
                    "dropout",
                    "weight_decay",
                ]:
                    self.issues.append(
                        f"hpo.yaml: Search parameter '{param}' not found in "
                        f"model or training configs"
                    )

    def print_report(self):
        """Print validation report."""
        is_valid, issues = self.validate_all()

        print("\n" + "=" * 60)
        print("AstroLab Configuration Validation Report")
        print("=" * 60)

        if is_valid:
            print("\n✅ All configuration files are valid!")
        else:
            print(f"\n❌ Found {len(issues)} issues:\n")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")

        print("\n" + "=" * 60)

        # Print config summary
        print("\n📋 Configuration Summary:")
        print(f"  Surveys: {len(self.configs.get('surveys', {}))}")
        print(f"  Tasks: {len(self.configs.get('tasks', {}))}")
        print(
            f"  Data directory: {self.configs.get('data', {}).get('base_dir', 'not set')}"
        )
        print(
            f"  Default model: {self.configs.get('model', {}).get('conv_type', 'not set')}"
        )
        print(
            f"  Default epochs: {self.configs.get('training', {}).get('max_epochs', 'not set')}"
        )


# CLI command for validation
if __name__ == "__main__":
    validator = ConfigValidator()
    validator.print_report()
