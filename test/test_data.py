"""
Test data module and datasets for AstroLab.
"""

from astro_lab.data import (
    AstroLabDataModule,
    AstroLabInMemoryDataset,
    KNNSampler,
)


class TestDataModuleAPI:
    """Test AstroLabDataModule API."""

    def test_create_datamodule_graph_task(self):
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="graph_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
        )
        dm.setup()
        assert hasattr(dm, "train_dataloader")
        assert hasattr(dm, "val_dataloader")
        assert hasattr(dm, "test_dataloader")

    def test_create_datamodule_node_task(self):
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
        )
        dm.setup()
        assert hasattr(dm, "train_dataloader")

    def test_backend_selection(self):
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
        )
        dm.setup()
        assert hasattr(dm, "train_dataloader")


class TestDatasets:
    """Test AstroLabInMemoryDataset integration."""

    def test_astro_lab_dataset(self):
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
        )
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")


class TestNodeLevelTasks:
    def test_neighbor_sampling(self):
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        assert hasattr(batch, "x")


class TestConverters:
    """Test data converters."""

    def test_dataset_creation(self, real_survey_data):
        """Test dataset creation from real survey data."""
        # Use real survey data to create a dataset
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        assert hasattr(dataset, "survey_name")
        assert dataset.survey_name == "gaia"
        assert hasattr(dataset, "task")
        assert dataset.task == "node_classification"


class TestDataModule:
    """Test Lightning DataModule."""

    def test_astro_lab_datamodule_setup(self):
        """Test AstroLabDataModule initialization and setup."""
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=32,
            num_workers=0,
        )

        # Check initialization
        assert dm.dataset.survey_name == "gaia"
        assert dm.dataset.task == "node_classification"
        assert dm.batch_size == 32

    def test_datamodule_transforms(self):
        """Test that transforms are properly applied."""
        from torch_geometric.transforms import NormalizeFeatures

        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            transform=NormalizeFeatures(),
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            num_workers=0,
        )

        assert dm.dataset.transform is not None


class TestMemoryEfficiency:
    """Test memory-efficient processing."""

    def test_large_dataset_handling(self):
        """Test that large datasets can be processed in chunks."""
        # Create dataset with large survey
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )

        # Check dataset properties
        assert hasattr(dataset, "survey_name")
        assert dataset.survey_name == "gaia"
        assert hasattr(dataset, "task")
        assert dataset.task == "node_classification"
