#!/usr/bin/env python3
"""AstroLab Configuration CLI - Show, create and validate configuration files."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from ..config import (
    get_config,
    get_data_config,
    get_model_config,
    get_survey_config,
    get_task_config,
    get_training_config,
)
from ..config_validator import ConfigValidator


def show_surveys():
    """Show all available surveys and their configurations."""
    config = get_config()
    surveys = config.get("surveys", {})

    print(f"\n{'=' * 80}")
    print(f"{'Survey':<15} {'Name':<30} {'Model':<10} {'Features':<20}")
    print(f"{'=' * 80}")

    for survey_key, survey_config in surveys.items():
        name = survey_config.get("name", "N/A")

        # Get recommended model
        rec_model = survey_config.get("recommended_model", {})
        model_type = rec_model.get("conv_type", "default")

        # Count features
        n_features = (
            len(survey_config.get("coord_cols", []))
            + len(survey_config.get("mag_cols", []))
            + len(survey_config.get("extra_cols", []))
            + len(survey_config.get("color_pairs", []))
        )

        print(f"{survey_key:<15} {name:<30} {model_type:<10} {n_features:<20}")

    print("\n💡 To see details: astro-lab config show <survey>")
    print("💡 To validate: astro-lab config validate")


def show_survey_config(survey: str):
    """Show detailed configuration for a specific survey."""
    try:
        config = get_survey_config(survey)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print(f"\n{'=' * 60}")
    print(f"Configuration for {survey.upper()}")
    print(f"{'=' * 60}\n")

    # Basic info
    print("📋 Basic Information:")
    print(f"  Name: {config.get('name', 'N/A')}")
    print(f"  Data Release: {config.get('data_release', 'N/A')}")
    print(f"  Coordinate System: {config.get('coordinate_system', 'N/A')}")
    print(f"  Filter System: {config.get('filter_system', 'N/A')}")

    # Columns
    print("\n🌐 Data Columns:")
    print(f"  Coordinates: {', '.join(config.get('coord_cols', []))}")
    print(f"  Magnitudes: {', '.join(config.get('mag_cols', []))}")
    print(f"  Extra: {', '.join(config.get('extra_cols', []))}")

    # Color pairs
    if config.get("color_pairs"):
        print("\n🎨 Color Indices:")
        for pair in config["color_pairs"]:
            print(f"    {pair[0]} - {pair[1]}")

    # Model recommendation
    if "recommended_model" in config:
        rec = config["recommended_model"]
        print("\n🧠 Recommended Model:")
        print(f"  Type: {rec.get('conv_type', 'N/A')}")
        print(f"  Layers: {rec.get('num_layers', 'N/A')}")
        print(f"  Hidden Dim: {rec.get('hidden_dim', 'N/A')}")
        if "heads" in rec:
            print(f"  Attention Heads: {rec['heads']}")
        print(f"  Dropout: {rec.get('dropout', 'N/A')}")

    # Training parameters
    print("\n⚙️  Training Parameters:")
    print(f"  Batch Size: {config.get('batch_size', 'default')}")
    print(f"  K-Neighbors: {config.get('k_neighbors', 'default')}")
    print(f"  Precision: {config.get('precision', 'default')}")
    print(f"  Experiment: {config.get('experiment_name', 'default')}")


def create_config(survey: str, task: str, output: Path, template: Optional[str] = None):
    """Create a custom configuration file for a survey and task."""
    try:
        survey_config = get_survey_config(survey)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    # Get task config
    try:
        task_config = get_task_config(task)
    except KeyError:
        print(f"❌ Error: Unknown task '{task}'")
        return

    # Get base configs
    data_config = get_data_config()
    model_config = get_model_config()
    training_config = get_training_config()

    # Start with template or defaults
    if template and Path(template).exists():
        with open(template, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Build comprehensive config
    config.update(
        {
            "survey": survey,
            "task": task,
            "data": {
                "batch_size": survey_config.get(
                    "batch_size", data_config.get("batch_size", 32)
                ),
                "k_neighbors": survey_config.get(
                    "k_neighbors", data_config.get("k_neighbors", 20)
                ),
                "num_workers": data_config.get("num_workers", 4),
                "sampling_strategy": data_config.get("sampling_strategy", "knn"),
            },
            "model": survey_config.get(
                "recommended_model",
                {
                    "conv_type": task_config.get(
                        "conv_type", model_config.get("conv_type", "gcn")
                    ),
                    "hidden_dim": model_config.get("hidden_dim", 128),
                    "num_layers": model_config.get("num_layers", 3),
                    "dropout": model_config.get("dropout", 0.1),
                    "pooling": task_config.get(
                        "pooling", model_config.get("pooling", "mean")
                    ),
                },
            ),
            "training": {
                "max_epochs": training_config.get("max_epochs", 100),
                "learning_rate": training_config.get("learning_rate", 0.001),
                "weight_decay": training_config.get("weight_decay", 0.0001),
                "optimizer": training_config.get("optimizer", "adamw"),
                "scheduler": training_config.get("scheduler", "cosine"),
                "precision": survey_config.get(
                    "precision", training_config.get("precision", "16-mixed")
                ),
                "gradient_clip_val": survey_config.get(
                    "gradient_clip_val", training_config.get("gradient_clip_val", 1.0)
                ),
            },
            "experiment": {
                "name": survey_config.get(
                    "experiment_name", f"{survey}_{task}_experiments"
                ),
                "run_name": f"{survey}_{task}_{config['model']['conv_type']}",
            },
        }
    )

    # Write config file
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created configuration file: {output}")
    print("\n📋 Configuration Summary:")
    print(f"  Survey: {survey}")
    print(f"  Task: {task}")
    print(f"  Model: {config['model']['conv_type']}")
    print(f"  Batch Size: {config['data']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"\n💡 To train: astro-lab train {survey} -c {output}")


def validate_config(fix: bool = False):
    """Validate all configuration files."""
    validator = ConfigValidator()
    is_valid, issues = validator.validate_all()

    print("\n" + "=" * 60)
    print("AstroLab Configuration Validation")
    print("=" * 60)

    if is_valid:
        print("\n✅ All configuration files are valid!")
    else:
        print(f"\n❌ Found {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        if fix:
            print("\n🔧 Attempting to fix issues...")
            # Here we could implement auto-fix logic
            print("⚠️  Auto-fix not yet implemented. Please fix manually.")

    print("\n" + "=" * 60)
    return 0 if is_valid else 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for config CLI."""
    parser = argparse.ArgumentParser(
        description="Manage AstroLab configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Show surveys command
    subparsers.add_parser("surveys", help="Show all available surveys")

    # Show specific config
    show_parser = subparsers.add_parser("show", help="Show configuration details")
    show_parser.add_argument("survey", help="Survey name (e.g., gaia, sdss)")

    # Create config command
    create_parser = subparsers.add_parser(
        "create", help="Create custom configuration file"
    )
    create_parser.add_argument("survey", help="Survey name")
    create_parser.add_argument("task", help="Task type (e.g., node_classification)")
    create_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output configuration file path",
    )
    create_parser.add_argument(
        "-t", "--template", type=Path, help="Template configuration file"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration files"
    )
    validate_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix issues automatically"
    )

    return parser


def main(args=None) -> int:
    """Main function for config CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == "surveys":
            show_surveys()
            return 0

        elif parsed_args.command == "show":
            show_survey_config(parsed_args.survey)
            return 0

        elif parsed_args.command == "create":
            create_config(
                parsed_args.survey,
                parsed_args.task,
                parsed_args.output,
                parsed_args.template,
            )
            return 0

        elif parsed_args.command == "validate":
            return validate_config(parsed_args.fix)

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
