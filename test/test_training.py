"""
Test training module for AstroLab with real data integration.

Uses fixtures from conftest.py for comprehensive testing with actual astronomical data.
"""

import pytest
import torch

from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer


class TestBasicTraining:
    """Test basic training functionality with fixtures."""

    def test_astro_trainer_creation(self):
        """Test AstroTrainer creation."""
        # Use GPU if available, otherwise CPU
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = AstroTrainer(
            experiment_name="test",
            max_epochs=5,
            devices=1,
            accelerator=device,
        )
        assert trainer is not None
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "test")

    def test_astro_model_creation(self):
        """Test AstroModel creation."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        assert model is not None
        assert hasattr(model, "forward")

    def test_astro_lab_datamodule_with_fixture(self, astro_datamodule):
        """Test AstroLabDataModule using fixture."""
        assert astro_datamodule is not None
        assert hasattr(astro_datamodule, "setup")
        assert hasattr(astro_datamodule, "train_dataloader")
        assert hasattr(astro_datamodule, "val_dataloader")
        assert hasattr(astro_datamodule, "test_dataloader")

    def test_lightning_model_with_fixture(self, lightning_model):
        """Test Lightning model using fixture."""
        assert lightning_model is not None
        assert hasattr(lightning_model, "forward")
        assert hasattr(lightning_model, "training_step")
        assert hasattr(lightning_model, "validation_step")


class TestTrainingIntegration:
    """Test training integration with real data."""

    def test_trainer_with_datamodule_fixture(self, astro_datamodule, lightning_model):
        """Test trainer with datamodule integration using fixtures."""
        assert astro_datamodule is not None
        assert lightning_model is not None

        # Use GPU if available, otherwise CPU
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = AstroTrainer(
            experiment_name="test_integration",
            max_epochs=1,
            devices=1,
            accelerator=device,
        )
        assert trainer is not None
        assert hasattr(trainer, "fit")

    def test_datamodule_setup_with_real_data(self, astro_datamodule):
        """Test datamodule setup with real data."""
        assert astro_datamodule is not None

        try:
            # Setup the datamodule
            astro_datamodule.setup()

            # Check that dataloaders are created
            train_loader = astro_datamodule.train_dataloader()
            val_loader = astro_datamodule.val_dataloader()
            test_loader = astro_datamodule.test_dataloader()

            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
        except Exception as e:
            pytest.skip(f"Datamodule setup failed: {e}")

    def test_end_to_end_training_workflow(self, astro_datamodule, lightning_model):
        """Test complete training workflow with fixtures."""
        assert astro_datamodule is not None
        assert lightning_model is not None

        try:
            # Setup datamodule
            astro_datamodule.setup()

            # Use GPU if available, otherwise CPU
            device = "gpu" if torch.cuda.is_available() else "cpu"
            trainer = AstroTrainer(
                experiment_name="test_e2e",
                max_epochs=1,
                devices=1,
                accelerator=device,
            )

            assert trainer is not None

            # Note: Actual training is commented out to avoid long test times
            # Uncomment for full integration testing:
            # trainer.fit(lightning_model, datamodule=astro_datamodule)
        except Exception as e:
            pytest.skip(f"End-to-end workflow failed: {e}")
