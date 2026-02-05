#!/usr/bin/env python3
"""
AstroLab Hyperparameter Optimization CLI
========================================

Optimized HPO using Optuna with efficient memory management for 2025.
Specifically optimized for RTX 4070 and similar GPUs.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import lightning as L
import optuna
import torch
import yaml
from optuna.integration import PyTorchLightningPruningCallback

# from ...data.datamodules.lightning import create_lightning_datamodule
# from ...models.core.factory import create_model
# from ...training.trainer_utils import setup_torch_compile


# Temporary placeholders - these functions need to be implemented
def create_lightning_datamodule(*args, **kwargs):
    """Placeholder for create_lightning_datamodule"""
    raise NotImplementedError("create_lightning_datamodule not implemented yet")


def create_model(*args, **kwargs):
    """Placeholder for create_model"""
    raise NotImplementedError("create_model not implemented yet")


def setup_torch_compile(*args, **kwargs):
    """Placeholder for setup_torch_compile"""
    raise NotImplementedError("setup_torch_compile not implemented yet")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for HPO CLI."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for AstroLab models using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  astro-lab hpo config.yaml --trials 50 --timeout 3600
  
  # Multi-GPU optimization with pruning
  astro-lab hpo config.yaml --trials 100 --devices 2 --pruner hyperband
  
  # Quick optimization for debugging
  astro-lab hpo config.yaml --trials 10 --max-samples 1000
  
  # Production optimization with database
  astro-lab hpo config.yaml --study-name gaia_hpo --storage sqlite:///optuna.db
  
  # Resume previous study
  astro-lab hpo config.yaml --study-name gaia_hpo --storage sqlite:///optuna.db --resume
""",
    )

    parser.add_argument("config", type=Path, help="Path to base configuration file")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, help="Name for the Optuna study")
    parser.add_argument("--storage", type=str, help="Database URL for persistence")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default="maximize",
        help="Optimization direction",
    )
    parser.add_argument(
        "--metric", type=str, default="val_acc", help="Metric to optimize"
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "hyperband", "none"],
        default="median",
        help="Pruning algorithm",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--max-samples", type=int, help="Limit samples for debugging")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser


class EfficientObjective:
    """
    Efficient HPO objective with model recycling.

    Features:
    - One-time model compilation
    - Parameter reset between trials
    - Memory-efficient model reuse
    """

    def __init__(self, base_config: Dict[str, Any], args: argparse.Namespace):
        self.base_config = base_config
        self.args = args
        self.logger = setup_logging(args.verbose)

        # Model recycling
        self.model = None
        self.compiled_model = None
        self.datamodule = None
        self.best_score = (
            float("inf") if args.direction == "minimize" else float("-inf")
        )
        self.best_params = None

        # Adaptive configuration
        self.adaptive_config = self._get_adaptive_config()

        # One-time compilation flag
        self.model_compiled = False

        self.logger.info("Efficient HPO objective initialized with model recycling")

    def _get_adaptive_config(self) -> Dict[str, Any]:
        """Get adaptive configuration based on hardware."""
        config = {}

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

            # RTX 4070 specific optimizations
            if "4070" in gpu_name:
                self.logger.info("Detected RTX 4070 - enabling specific optimizations")
                config.update(
                    {
                        "use_torch_compile": True,
                        "compile_mode": "default",
                        "precision": "16-mixed",
                        "gradient_checkpointing": True,
                    }
                )

        return config

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute one optimization trial with model recycling."""
        try:
            # Suggest hyperparameters
            hparams = self._suggest_hyperparameters(trial)

            # Merge configurations
            config = {**self.base_config, **self.adaptive_config, **hparams}

            # Apply debugging overrides
            if self.args.max_samples:
                config["max_samples"] = self.args.max_samples

            self.logger.info(f"Trial {trial.number}: {hparams}")

            # Setup data module (reuse if possible)
            if self.datamodule is None or config["batch_size"] != self.base_config.get(
                "batch_size"
            ):
                self.datamodule = self._create_datamodule(config)

            # Create or recycle model
            if self.model is None:
                # First trial - create model
                self.model = self._create_model(config)
                self.logger.info("Created new model for first trial")
            else:
                # Recycle existing model - reset parameters only
                self._reset_model_parameters(config)
                self.logger.info("Recycled existing model with parameter reset")

            # One-time compilation if enabled
            if (
                not self.model_compiled
                and config.get("use_torch_compile", False)
                and torch.cuda.is_available()
            ):
                self._compile_model_once()

            # Train model
            score = self._train_model(trial, config)

            # Track best
            if self._is_better(score):
                self.best_score = score
                self.best_params = hparams.copy()
                self._save_checkpoint()

            # Cleanup
            self._cleanup()

            return score

        except optuna.TrialPruned:
            self.logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            if self.args.verbose:
                import traceback

                traceback.print_exc()
            # Return worst possible score
            return float("inf") if self.args.direction == "minimize" else float("-inf")

    def _compile_model_once(self):
        """Compile model once for all trials using central compilation."""
        try:
            if self.model is None:
                return

            device = next(self.model.parameters()).device

            self.compiled_model = setup_torch_compile(
                model=self.model,
                device=device,
                use_torch_compile=True,
                compile_mode=self.adaptive_config.get("compile_mode", "default"),
                compile_dynamic=True,
            )

            # Replace model with compiled version if compilation succeeded
            if self.compiled_model is not self.model:
                self.model = self.compiled_model
                self.model_compiled = True
                self.logger.info("Model compiled successfully for all trials")
            else:
                self.logger.info("Model compilation not applied (using original)")

        except Exception as e:
            self.logger.warning(
                f"Model compilation failed: {e}. Using uncompiled version."
            )

    def _reset_model_parameters(self, config: Dict[str, Any]) -> None:
        """Reset model parameters using existing mixin methods."""
        # Use the existing HPOResetMixin methods
        self.model.reset_all_parameters()
        self.model.reset_optimizer_states()
        self.model.cleanup_memory()

        # Update hyperparameters
        if hasattr(self.model, "hparams"):
            self.model.hparams.update(config)

        # Update learning rate and weight decay
        if hasattr(self.model, "learning_rate"):
            self.model.learning_rate = config["learning_rate"]
        if hasattr(self.model, "weight_decay"):
            self.model.weight_decay = config["weight_decay"]

        self.logger.info("Model parameters reset using mixin methods")

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters optimized for GNNs."""
        hparams = {}

        # Architecture
        hparams["conv_type"] = trial.suggest_categorical(
            "conv_type", ["gcn", "gat", "sage"]
        )
        hparams["num_layers"] = trial.suggest_int("num_layers", 2, 4)
        hparams["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])

        # Conv-specific
        if hparams["conv_type"] == "gat":
            hparams["heads"] = trial.suggest_categorical("heads", [2, 4, 8])
            hparams["attention_dropout"] = trial.suggest_float(
                "attention_dropout", 0.0, 0.3
            )
        elif hparams["conv_type"] == "sage":
            hparams["aggr"] = trial.suggest_categorical("aggr", ["mean", "max"])

        # Training
        hparams["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        hparams["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        )
        hparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        hparams["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])

        # Batch size based on memory
        if self.adaptive_config.get("gpu_memory_gb", 0) >= 12:  # RTX 4070 and up
            hparams["batch_size"] = trial.suggest_categorical(
                "batch_size", [32, 48, 64]
            )
        else:
            hparams["batch_size"] = trial.suggest_categorical(
                "batch_size", [16, 24, 32]
            )

        # Sampling strategy for large graphs
        if self.base_config.get("num_nodes", 0) > 100000:
            hparams["sampling_strategy"] = trial.suggest_categorical(
                "sampling_strategy", ["neighbor", "cluster"]
            )
            if hparams["sampling_strategy"] == "neighbor":
                hparams["neighbor_sizes"] = [
                    trial.suggest_int("neighbor_size_1", 10, 30),
                    trial.suggest_int("neighbor_size_2", 5, 15),
                ]

        return hparams

    def _create_datamodule(self, config: Dict[str, Any]) -> L.LightningDataModule:
        """Create data module with config."""
        return create_lightning_datamodule(
            survey=config["survey"],
            task=config.get("task", "node"),
            batch_size=config["batch_size"],
            num_workers=min(4, config.get("num_workers", 4)),
            max_samples=config.get("max_samples"),
            sampling_strategy=config.get("sampling_strategy", "none"),
            neighbor_sizes=config.get("neighbor_sizes", [25, 10]),
        )

    def _create_model(self, config: Dict[str, Any]) -> L.LightningModule:
        """Create new model with HPO support."""
        # Get dataset info
        if self.datamodule:
            self.datamodule.setup()
            config["num_features"] = self.datamodule.num_features
            config["num_classes"] = self.datamodule.num_classes

        model = create_model(
            model_type=config.get("model_type", "node"),
            task=config.get("task", "node_classification"),
            num_features=config.get("num_features", 128),
            num_classes=config.get("num_classes", 7),
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            conv_type=config["conv_type"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            # Enable optimizations
            use_torch_compile=config.get("use_torch_compile", False),
            compile_mode=config.get("compile_mode", "default"),
            compile_dynamic=True,
        )

        return model

    def _update_model(self, config: Dict[str, Any]) -> None:
        """Update existing model efficiently."""
        # Update hyperparameters
        self.model.hparams.update(config)

        # Reset architecture if needed
        if hasattr(self.model, "build_dynamic_architecture"):
            self.model.build_dynamic_architecture(config)

        # Reset parameters
        if hasattr(self.model, "reset_all_parameters"):
            self.model.reset_all_parameters()
        else:
            # Manual reset
            for module in self.model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        # Update learning rate
        self.model.learning_rate = config["learning_rate"]
        self.model.weight_decay = config["weight_decay"]

    def _train_model(self, trial: optuna.Trial, config: Dict[str, Any]) -> float:
        """Train model and return score."""
        # Callbacks
        callbacks = []

        # Pruning callback
        if self.args.pruner != "none":
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor=self.args.metric)
            )

        # Early stopping
        callbacks.append(
            L.callbacks.EarlyStopping(
                monitor=self.args.metric,
                patience=5,
                mode="max" if self.args.direction == "maximize" else "min",
            )
        )

        # Model checkpoint (optional)
        if config.get("save_checkpoints", False):
            callbacks.append(
                L.callbacks.ModelCheckpoint(
                    monitor=self.args.metric,
                    mode="max" if self.args.direction == "maximize" else "min",
                    save_top_k=1,
                )
            )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=min(20, config.get("max_epochs", 50)),  # Shorter for HPO
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.args.devices,
            precision=config.get("precision", "16-mixed"),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,  # Disable for HPO
            gradient_clip_val=config.get("gradient_clip_val", 1.0),
            accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
            # Optimizations
            benchmark=True,  # Enable cuDNN benchmark
            deterministic=False,  # Faster but non-deterministic
        )

        # Train
        trainer.fit(self.model, self.datamodule)

        # Get score
        if self.args.metric in trainer.callback_metrics:
            return float(trainer.callback_metrics[self.args.metric])
        else:
            # Fallback
            return float("inf") if self.args.direction == "minimize" else float("-inf")

    def _is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.args.direction == "maximize":
            return score > self.best_score
        else:
            return score < self.best_score

    def _save_checkpoint(self) -> None:
        """Save best model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "best_score": self.best_score,
            "best_params": self.best_params,
        }
        torch.save(checkpoint, "best_hpo_model.pt")

        # Also save config
        with open("best_hpo_config.yaml", "w") as f:
            yaml.dump(self.best_params, f)

    def _cleanup(self) -> None:
        """Clean up memory after trial."""
        # Clear caches
        if hasattr(self.model, "cleanup_memory"):
            self.model.cleanup_memory()

        # Garbage collection
        import gc

        gc.collect()

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def create_study(args: argparse.Namespace) -> optuna.Study:
    """Create or load Optuna study."""
    # Pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=5,
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Sampler - TPE with multivariate
    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,  # Consider correlations
        group=True,  # Group related parameters
    )

    # Create study
    study = optuna.create_study(
        direction=args.direction,
        pruner=pruner,
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
    )

    return study


def main(args=None) -> int:
    """Main HPO function."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    logger = setup_logging(parsed_args.verbose)

    # Load base config
    with open(parsed_args.config, "r") as f:
        base_config = yaml.safe_load(f)

    logger.info(f"Loaded config from {parsed_args.config}")

    # Create study
    study = create_study(parsed_args)

    # Log study info
    logger.info(f"Starting HPO with {parsed_args.trials} trials")
    logger.info(f"Direction: {parsed_args.direction} {parsed_args.metric}")

    # Create objective
    objective = EfficientObjective(base_config, parsed_args)

    try:
        # Run optimization
        study.optimize(
            objective,
            n_trials=parsed_args.trials,
            timeout=parsed_args.timeout,
            gc_after_trial=True,  # Force GC after each trial
            show_progress_bar=True,
        )

        # Results
        logger.info("\nOptimization completed!")
        logger.info(f"Finished trials: {len(study.trials)}")

        if study.best_trial:
            logger.info(f"\nBest trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_value:.4f}")
            logger.info("\nBest parameters:")
            for key, value in study.best_params.items():
                logger.info(f"  {key}: {value}")

            # Save results
            results = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial": study.best_trial.number,
                "n_trials": len(study.trials),
            }

            with open("hpo_results.yaml", "w") as f:
                yaml.dump(results, f, default_flow_style=False)

            logger.info("\nResults saved to hpo_results.yaml")

            # Optionally create visualization
            import optuna.visualization as vis

            # Parameter importance
            fig = vis.plot_param_importances(study)
            fig.write_html("param_importance.html")

            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html("optimization_history.html")

            logger.info("Visualizations saved to HTML files")

        return 0

    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted")
        return 1
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if parsed_args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
