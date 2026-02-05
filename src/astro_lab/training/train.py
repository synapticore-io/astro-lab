"""Simple training script for AstroLab models."""

import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from astro_lab.config import (
    get_combined_config,
    get_model_config,
    get_survey_config,
    get_task_config,
)
from astro_lab.config_validator import ConfigValidator
from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from astro_lab.data.dataset.lightning import AstroLabDataModule
from astro_lab.data.samplers.neighbor import KNNSampler
from astro_lab.models import AstroModel
from astro_lab.training.trainer import AstroTrainer
from astro_lab.utils.device import is_cuda_available

logger = logging.getLogger(__name__)


def prepare_datamodule(
    survey: str,
    task: str,
    config: dict,
) -> Tuple[Any, dict]:
    """Prepare data module with all config parameters."""
    # Extract all data config parameters
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))  # Default to 0 for Windows
    max_samples = config.get("max_samples", None)

    # Sampling strategy parameters
    sampling_strategy = config.get("sampling_strategy", "knn")
    k_neighbors = int(config.get("k_neighbors", 8))
    neighbor_sizes = config.get("neighbor_sizes", [25, 10])
    num_clusters = int(config.get("num_clusters", 1500))
    saint_sample_coverage = int(config.get("saint_sample_coverage", 50))
    enable_dynamic_batching = config.get("enable_dynamic_batching", False)

    # Log all data parameters
    logger.info("Data configuration:")
    logger.info(f"  Sampling strategy: {sampling_strategy}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  K-neighbors: {k_neighbors}")
    logger.info(f"  Neighbor sizes: {neighbor_sizes}")
    logger.info(f"  Num clusters: {num_clusters}")
    logger.info(f"  SAINT coverage: {saint_sample_coverage}")
    logger.info(f"  Dynamic batching: {enable_dynamic_batching}")

    try:
        # Create dataset with sampling configuration
        sampler_kwargs = {
            "k": k_neighbors,
            "neighbor_sizes": neighbor_sizes,
            "num_clusters": num_clusters,
            "saint_sample_coverage": saint_sample_coverage,
        }

        dataset = AstroLabInMemoryDataset(
            survey_name=survey,
            sampling_strategy=sampling_strategy,
            sampler_kwargs=sampler_kwargs,
            task=task,
            max_samples=max_samples,
        )

        # Check if data exists
        processed_path = Path(dataset.processed_paths[0])
        if not processed_path.exists():
            logger.info(f"Processing dataset for {survey}...")
            try:
                dataset.process()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"\n{'=' * 60}\n"
                    f"ERROR: Could not find preprocessed data for survey '{survey}'!\n"
                    f"{'=' * 60}\n\n"
                    f"Please run the following commands first:\n\n"
                    f"1. Download the raw data:\n"
                    f"   astro-lab download {survey}\n\n"
                    f"2. Preprocess the data:\n"
                    f"   astro-lab preprocess {survey}\n\n"
                    f"Original error: {e}\n"
                    f"{'=' * 60}\n"
                )

        # Create appropriate sampler based on strategy
        if sampling_strategy == "knn":
            sampler = KNNSampler(k=k_neighbors)
        elif sampling_strategy == "neighbor":
            # Import and use NeighborSampler with neighbor_sizes

            # This will be handled in the DataModule
            sampler = None
        else:
            # Default sampler
            sampler = KNNSampler(k=k_neighbors)

        # Create datamodule with all parameters
        datamodule = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            enable_dynamic_batching=enable_dynamic_batching,
            neighbor_sizes=neighbor_sizes if sampling_strategy == "neighbor" else None,
        )

        # Setup datamodule
        datamodule.setup()

        # Get dataset info
        info = dataset.get_info()

        # Validate required info
        if "num_features" not in info:
            raise ValueError(f"Dataset info missing 'num_features': {info}")
        if "num_classes" not in info:
            raise ValueError(f"Dataset info missing 'num_classes': {info}")

        logger.info("Dataset loaded successfully:")
        logger.info(f"  Survey: {survey}")
        logger.info(f"  Task: {task}")
        logger.info(f"  Samples: {info.get('num_samples', 'unknown')}")
        logger.info(f"  Features: {info['num_features']}")
        logger.info(f"  Classes: {info['num_classes']}")

        return datamodule, info

    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise  # Re-raise FileNotFoundError with our custom message
        logger.error(f"Failed to prepare datamodule for survey '{survey}': {e}")
        raise


def prepare_model(
    survey: str,
    task: str,
    model_type: Optional[str],
    info: dict,
    config: dict,
) -> Any:
    """Prepare model with proper configuration and survey awareness."""

    # Get survey-specific config
    survey_config = get_survey_config(survey)
    recommended_model = survey_config.get("recommended_model", {})

    # If model_type is "auto" or not specified, use recommendation
    if not model_type or model_type == "auto":
        model_type = recommended_model.get("conv_type", "gcn")
        logger.info(f"Using recommended model type '{model_type}' for {survey}")

    # Build model configuration with priority:
    # 1. CLI/config overrides (highest)
    # 2. Survey recommendations
    # 3. Task defaults
    # 4. General model defaults

    # Start with general model defaults
    model_config = get_model_config()

    # Apply task-specific defaults
    task_config = get_task_config(task)
    if task_config.get("conv_type") and not config.get("conv_type"):
        model_config["conv_type"] = task_config["conv_type"]
    if task_config.get("pooling"):
        model_config["pooling"] = task_config["pooling"]

    # Apply survey recommendations if using recommended model type
    if model_type == recommended_model.get("conv_type"):
        model_config.update(recommended_model)

    # Check model-task compatibility
    if task == "graph_classification" and model_config.get("pooling") is None:
        logger.warning(
            f"Survey {survey} recommends {model_type} without pooling "
            f"for graph task - adding default pooling='mean'"
        )
        model_config["pooling"] = "mean"

    # Apply user config overrides
    for key in [
        "hidden_dim",
        "num_layers",
        "dropout",
        "learning_rate",
        "heads",
        "pooling",
        "activation",
        "norm",
        "residual",
        "edge_dim",
        "warmup_epochs",
    ]:
        if key in config:
            model_config[key] = config[key]

    # Set required parameters
    model_config.update(
        {
            "num_features": info["num_features"],
            "num_classes": info["num_classes"],
            "conv_type": model_type,
            "task": task,
            "learning_rate": float(model_config.get("learning_rate", 1e-3)),
            "weight_decay": float(config.get("weight_decay", 1e-4)),
        }
    )

    # Log final configuration
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Model Configuration for {survey}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Survey: {survey}")
    logger.info(f"  Coordinate columns: {survey_config.get('coord_cols', [])}")
    logger.info(f"  Magnitude columns: {survey_config.get('mag_cols', [])}")
    logger.info(f"  Extra features: {survey_config.get('extra_cols', [])}")
    logger.info(f"Model: {model_type}")
    logger.info("  Architecture: AstroModel")
    logger.info(f"  Input features: {model_config['num_features']}")
    logger.info(f"  Output classes: {model_config['num_classes']}")
    logger.info(f"  Hidden dimension: {model_config.get('hidden_dim', 128)}")
    logger.info(f"  Number of layers: {model_config.get('num_layers', 3)}")
    if model_type in ["gat", "transformer"] and "heads" in model_config:
        logger.info(f"  Attention heads: {model_config['heads']}")
    if "edge_dim" in model_config and model_config["edge_dim"]:
        logger.info(f"  Edge features: {model_config['edge_dim']}")
    logger.info(f"{'=' * 60}\n")

    # Check for HeteroData metadata
    metadata = None
    if (
        info.get("pyg_type") == "HeteroData"
        and "node_types" in info
        and "edge_types" in info
    ):
        metadata = (info["node_types"], info["edge_types"])
        model_config["metadata"] = metadata

    # Create model
    return AstroModel(**model_config)


def train_model(
    survey: str,
    task: str = "node_classification",
    model_type: Optional[str] = None,
    run_name: str = "",
    config: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Train an AstroLab model with proper error handling and validation."""

    # Defensive: convert Namespace to dict if needed
    if config is not None and hasattr(config, "__dict__"):
        config = vars(config)
    if config is None:
        config = get_combined_config(survey, task)

    # Validate configuration before training
    validator = ConfigValidator()
    is_valid, issues = validator.validate_all()
    if not is_valid:
        logger.warning(f"Config validation found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5 issues
            logger.warning(f"  - {issue}")
        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more issues")

    # Cache CUDA availability check
    cuda_available = is_cuda_available()

    # Extract all config parameters with defaults in one place
    config_defaults = {
        "batch_size": 32,
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "checkpoint_monitor": "val_loss",
        "checkpoint_save_top_k": 3,
        "val_check_interval": 1.0,
        "limit_train_batches": 1.0,
        "enable_model_summary": True,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "experiment_name": "astro_gnn",
        "devices": 1 if cuda_available else "cpu",
        "accelerator": "gpu" if cuda_available else "cpu",
        "precision": "32-true",
    }

    # Get config values with defaults
    model_type = model_type or config.get("conv_type") or config.get("model_type")
    experiment_name = str(
        config.get("experiment_name", config_defaults["experiment_name"])
    )
    devices = config.get("devices", config_defaults["devices"])
    accelerator = str(config.get("accelerator", config_defaults["accelerator"]))
    gradient_clip_val = float(
        config.get("gradient_clip_val", config_defaults["gradient_clip_val"])
    )
    max_epochs = int(config.get("max_epochs", config_defaults["max_epochs"]))
    accumulate_grad_batches = int(
        config.get(
            "accumulate_grad_batches", config_defaults["accumulate_grad_batches"]
        )
    )
    early_stopping = config.get("early_stopping", config_defaults["early_stopping"])
    early_stopping_patience = int(
        config.get(
            "early_stopping_patience", config_defaults["early_stopping_patience"]
        )
    )
    checkpoint_monitor = config.get(
        "checkpoint_monitor", config_defaults["checkpoint_monitor"]
    )
    checkpoint_save_top_k = int(
        config.get("checkpoint_save_top_k", config_defaults["checkpoint_save_top_k"])
    )
    val_check_interval = float(
        config.get("val_check_interval", config_defaults["val_check_interval"])
    )
    limit_train_batches = float(
        config.get("limit_train_batches", config_defaults["limit_train_batches"])
    )
    enable_model_summary = config.get(
        "enable_model_summary", config_defaults["enable_model_summary"]
    )
    scheduler = config.get("scheduler", config_defaults["scheduler"])
    warmup_epochs = int(config.get("warmup_epochs", config_defaults["warmup_epochs"]))

    # Set PyTorch optimizations
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting training for {survey} - {task}")
    logger.info(f"{'=' * 60}\n")

    # Prepare data
    try:
        datamodule, info = prepare_datamodule(
            survey=survey,
            task=task,
            config=config,
        )
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise

    # Validate data by checking a sample batch
    try:
        sample_batch = next(iter(datamodule.train_dataloader()))
        logger.info(f"Data validation passed - batch shape: {sample_batch}")
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise RuntimeError(
            "Could not load data from datamodule. "
            "Please check that the data was preprocessed correctly."
        ) from e

    # Prepare model
    try:
        model = prepare_model(
            survey=survey,
            task=task,
            model_type=model_type,
            info=info,
            config=config,
        )
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

    # Move model to GPU if available
    if cuda_available and accelerator == "gpu":
        model = model.cuda()
        logger.info("Model moved to GPU")

    # Optional model compilation (disabled on Windows)
    if config.get("compile_model", False) and platform.system() != "Windows":
        if hasattr(torch, "compile"):
            compile_mode = config.get("compile_mode", "default")
            try:
                model = torch.compile(model, mode=compile_mode, dynamic=True)
                logger.info(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Continuing without compilation."
                )

    # Create trainer with all config parameters
    trainer = AstroTrainer(
        experiment_name=experiment_name,
        run_name=run_name or None,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=str(config.get("precision", "32-true")),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_train_batches,
        enable_model_summary=enable_model_summary,
        enable_progress_bar=True,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        checkpoint_monitor=checkpoint_monitor,
        checkpoint_save_top_k=checkpoint_save_top_k,
        scheduler_config={
            "scheduler": scheduler,
            "warmup_epochs": warmup_epochs,
        },
        **kwargs,
    )

    logger.info("\nTraining configuration:")
    logger.info(f"  Model: {model_type}")
    logger.info(f"  Epochs: {max_epochs}")
    logger.info(f"  Batch size: {config.get('batch_size', 32)}")
    logger.info(f"  Learning rate: {config.get('learning_rate', 1e-3)}")
    logger.info(f"  Scheduler: {scheduler} (warmup: {warmup_epochs} epochs)")
    logger.info(f"  Device: {accelerator}")
    logger.info(f"  Precision: {config.get('precision', '32-true')}")
    logger.info(f"  Gradient accumulation: {accumulate_grad_batches}")

    # Train model
    try:
        logger.info("\nStarting training...")
        trainer.fit(model, datamodule)
        logger.info("Training completed!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Test model
    try:
        logger.info("\nRunning test evaluation...")
        test_results = trainer.test(model, datamodule)
        logger.info("Test evaluation completed!")
    except Exception as e:
        logger.warning(f"Test evaluation failed: {e}")
        test_results = None

    return {
        "model": model,
        "trainer": trainer,
        "test_results": test_results[0] if test_results else {},
        "info": info,
    }


def main():
    """Main function for standalone execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example training
    try:
        results = train_model(
            survey="gaia",
            task="node_classification",
            model_type=None,  # Use recommended model
            config={"max_epochs": 10, "max_samples": 1000},  # Quick test
        )
        print("\nTraining Results:")
        print(f"Test results: {results['test_results']}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    main()
