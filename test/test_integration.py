"""
Integration tests for AstroLab.
"""

import pytest
import torch

from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from astro_lab.data.dataset.lightning import AstroLabDataModule
from astro_lab.data.samplers.neighbor import KNNSampler
from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer


class TestEndToEnd:
    """Test complete pipelines from data to trained model."""

    @pytest.mark.slow
    def test_gaia_pipeline(self):
        """Test complete pipeline with GAIA data."""
        try:
            # Load small GAIA sample with limited data
            dataset = AstroLabInMemoryDataset(
                survey_name="gaia",
                task="node_classification",
                max_samples=100,  # Use only 100 samples for testing
            )

            # Get dataset info to determine actual feature dimensions
            info = dataset.get_info()

            sampler = KNNSampler(k=8)
            datamodule = AstroLabDataModule(
                dataset=dataset,
                sampler=sampler,
                batch_size=10,
                num_workers=0,  # Use 0 for compatibility
            )
            datamodule.setup()

            # Create model with actual feature dimensions
            model = AstroModel(
                num_features=info["num_features"],
                num_classes=info["num_classes"],
                hidden_dim=32,
                num_layers=2,
                task="node_classification",
            )
            assert model is not None
            assert datamodule is not None

        except (FileNotFoundError, RuntimeError) as e:
            if "No data found" in str(e) or "not been downloaded" in str(e):
                pytest.skip(
                    "GAIA data not available - run 'astro-lab download gaia' first"
                )
            else:
                raise

    @pytest.mark.slow
    def test_sdss_pipeline(self):
        """Test complete pipeline with SDSS data."""
        try:
            # Load small SDSS sample
            dataset = AstroLabInMemoryDataset(
                survey_name="sdss",
                task="node_classification",
                max_samples=100,
            )

            # Get dataset info
            info = dataset.get_info()

            sampler = KNNSampler(k=8)
            datamodule = AstroLabDataModule(
                dataset=dataset,
                sampler=sampler,
                batch_size=10,
                num_workers=0,
            )
            datamodule.setup()

            # Create model with actual dimensions
            model = AstroModel(
                num_features=info["num_features"],
                num_classes=info["num_classes"],
                hidden_dim=32,
                num_layers=2,
                task="node_classification",
            )

            # Create trainer with minimal epochs
            trainer = AstroTrainer(
                experiment_name="test_sdss_pipeline",
                max_epochs=1,  # Just 1 epoch for testing
                devices=1,
                accelerator="cpu",
                enable_progress_bar=False,
            )

            # Train model
            trainer.fit(model, datamodule)
            assert True  # If we get here, training succeeded

        except (FileNotFoundError, RuntimeError) as e:
            if "No data found" in str(e) or "not been downloaded" in str(e):
                pytest.skip(
                    "SDSS data not available - run 'astro-lab download sdss' first"
                )
            else:
                raise


class TestModelCreation:
    """Test model creation with different configurations."""

    def test_model_with_variable_features(self):
        """Test that model handles variable feature dimensions."""
        # Test different feature dimensions
        for num_features in [3, 10, 50, 128]:
            model = AstroModel(
                num_features=num_features,
                num_classes=6,
                hidden_dim=64,
                task="node_classification",
            )

            # Create dummy batch
            batch = type(
                "obj",
                (object,),
                {
                    "x": torch.randn(20, num_features),
                    "edge_index": torch.randint(0, 20, (2, 40)),
                    "num_nodes": 20,
                },
            )

            # Forward pass should work
            out = model(batch)
            assert out.shape == (20, 6)

    def test_model_device_handling(self):
        """Test that model handles device placement correctly."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=32,
            task="node_classification",
        )

        # Create batch on CPU
        batch = type(
            "obj",
            (object,),
            {
                "x": torch.randn(10, 10),
                "edge_index": torch.randint(0, 10, (2, 20)),
                "num_nodes": 10,
            },
        )

        # Model on CPU should work
        out = model(batch)
        assert out.device.type == "cpu"

        # If CUDA available, test GPU
        if torch.cuda.is_available():
            model = model.cuda()
            out = model(batch)  # Should handle CPU batch with GPU model
            assert out.device.type == "cuda"


class TestDataHandling:
    """Test data loading and processing."""

    def test_empty_dataset_error(self):
        """Test that empty dataset produces helpful error."""
        # This should fail with a clear error message
        with pytest.raises((RuntimeError, FileNotFoundError)) as excinfo:
            dataset = AstroLabInMemoryDataset(
                survey_name="nonexistent_survey",
                task="node_classification",
            )
            sampler = KNNSampler(k=8)
            datamodule = AstroLabDataModule(
                dataset=dataset,
                sampler=sampler,
                batch_size=32,
            )
            datamodule.setup()

        # Check that error message contains helpful instructions
        error_msg = str(excinfo.value)
        assert "astro-lab download" in error_msg or "No data found" in error_msg


class TestDataModelIntegration:
    """Test data and model integration."""

    def test_survey_model_compatibility(self):
        """Test that models work with different feature dimensions."""
        # Test various feature dimensions that surveys might have
        test_configs = [
            {"num_features": 3, "task": "node_classification"},  # Minimal coords
            {"num_features": 8, "task": "node_classification"},  # Basic features
            {"num_features": 15, "task": "node_classification"},  # Extended features
            {"num_features": 10, "task": "graph_classification"},
        ]

        for config in test_configs:
            try:
                # Create model
                model = AstroModel(
                    num_features=config["num_features"],
                    num_classes=3,
                    hidden_dim=32,
                    num_layers=2,
                    task=config["task"],
                )

                # Create appropriate dummy batch
                if config["task"] == "node_classification":
                    batch = type(
                        "obj",
                        (object,),
                        {
                            "x": torch.randn(20, config["num_features"]),
                            "edge_index": torch.randint(0, 20, (2, 40)),
                            "y": torch.randint(0, 3, (20,)),
                            "num_nodes": 20,
                        },
                    )
                else:  # graph_classification
                    batch = type(
                        "obj",
                        (object,),
                        {
                            "x": torch.randn(20, config["num_features"]),
                            "edge_index": torch.randint(0, 20, (2, 40)),
                            "y": torch.tensor([1]),  # Single graph label
                            "batch": torch.zeros(
                                20, dtype=torch.long
                            ),  # All nodes in same graph
                            "num_nodes": 20,
                        },
                    )

                # Test forward pass
                out = model(batch)
                assert out is not None

                if config["task"] == "node_classification":
                    assert out.shape[0] == 20  # Number of nodes
                else:
                    assert out.shape[0] == 1  # Single graph

                assert out.shape[1] == 3  # Number of classes

            except Exception as e:
                pytest.fail(f"Failed with config {config}: {e}")
