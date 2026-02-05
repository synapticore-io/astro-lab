"""
Test models module for AstroLab (Lightning API).
"""

import pytest
import torch
from torch_geometric.data import Data

from astro_lab.models import AstroModel
from src.astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from src.astro_lab.data.dataset.lightning import AstroLabDataModule


class TestModelCreation:
    """Test model creation."""

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

    def test_astro_model_with_different_tasks(self):
        """Test AstroModel with different tasks."""
        tasks = ["node_classification", "graph_classification", "node_regression"]

        for task in tasks:
            model = AstroModel(
                num_features=10,
                num_classes=3,
                hidden_dim=64,
                num_layers=2,
                task=task,
            )
            assert model is not None
            assert model.task == task


class TestModelFunctionality:
    """Test model forward passes and features."""

    def _get_nonempty_graph(self, dataset, indices):
        for idx in indices[:10]:  # Try first 10 indices
            graph = dataset.get(idx)
            if hasattr(graph, "edge_index") and graph.edge_index.size(1) > 0:
                return graph
        import pytest

        pytest.skip("No graph with edges found in first 10 indices.")

    def test_node_classification_forward(self):
        pass

        ds = AstroLabInMemoryDataset(survey_name="gaia", task="node_classification")
        dm = AstroLabDataModule(dataset=ds, batch_size=1, num_workers=0)
        dm.setup()
        graph = self._get_nonempty_graph(ds, dm.train_indices)
        from astro_lab.models import AstroModel

        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        out = model(graph)
        assert out.shape[0] == graph.x.shape[0]

    def test_graph_classification_forward(self):
        pass

        ds = AstroLabInMemoryDataset(survey_name="gaia", task="graph_classification")
        dm = AstroLabDataModule(dataset=ds, batch_size=1, num_workers=0)
        dm.setup()
        graph = self._get_nonempty_graph(ds, dm.train_indices)
        from astro_lab.models import AstroModel

        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="graph_classification",
        )
        out = model(graph)
        assert out.shape[1] == 3

    def test_node_regression_forward(self):
        pass

        ds = AstroLabInMemoryDataset(survey_name="gaia", task="node_regression")
        dm = AstroLabDataModule(dataset=ds, batch_size=1, num_workers=0)
        dm.setup()
        graph = self._get_nonempty_graph(ds, dm.train_indices)
        from astro_lab.models import AstroModel

        model = AstroModel(
            num_features=10,
            num_classes=1,  # Regression output
            hidden_dim=64,
            num_layers=2,
            task="node_regression",
        )
        out = model(graph)
        assert out.shape[0] == graph.x.shape[0]


class TestModelUtils:
    """Test model utilities."""

    def test_model_device_handling(self):
        """Test model device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        model = model.cuda()

        # Test forward on GPU
        x = torch.randn(10, 10).cuda()
        edge_index = torch.randint(0, 10, (2, 20)).cuda()
        batch = torch.zeros(10, dtype=torch.long).cuda()
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = model(data)
        assert out.is_cuda


class TestModelConfigs:
    """Test model configurations."""

    @pytest.mark.parametrize(
        "task",
        ["node_classification", "graph_classification", "node_regression"],
    )
    def test_model_configs(self, task):
        """Test each task with different configs."""
        # Small config
        model = AstroModel(
            num_features=8,
            num_classes=2,
            hidden_dim=16,
            num_layers=2,
            task=task,
        )
        assert model is not None

        # Medium config
        model = AstroModel(
            num_features=32,
            num_classes=5,
            hidden_dim=64,
            num_layers=3,
            task=task,
        )
        assert model is not None

        # Large config
        model = AstroModel(
            num_features=128,
            num_classes=10,
            hidden_dim=256,
            num_layers=4,
            task=task,
        )
        assert model is not None
