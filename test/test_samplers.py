"""Test all sampler types with the streaming dataset."""

import pytest

from src.astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset


@pytest.fixture
def astrolab_dataset():
    """Create a base AstroLabInMemoryDataset fixture."""
    return AstroLabInMemoryDataset(
        survey_name="gaia",
        task="node_classification",
        force_reload=True,
    )


class TestSamplers:
    """Test all sampler types with streaming dataset."""

    @pytest.fixture
    def dataset_knn(self):
        """Create dataset with KNNSampler."""
        return AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="knn",
            sampler_kwargs={"k": 2},
            force_reload=True,
        )

    @pytest.fixture
    def dataset_radius(self):
        """Create dataset with RadiusSampler."""
        return AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="radius",
            sampler_kwargs={"radius": 1.0, "max_num_neighbors": 10},
            force_reload=True,
        )

    @pytest.fixture
    def dataset_adaptive(self):
        """Create dataset with AdaptiveRadiusSampler."""
        return AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="adaptive",
            sampler_kwargs={
                "base_radius": 1.0,
                "target_neighbors": 5,
                "max_num_neighbors": 10,
            },
            force_reload=True,
        )

    @pytest.fixture
    def dataset_neighbor(self):
        """Create dataset with NeighborSubgraphSampler."""
        return AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="neighbor",
            sampler_kwargs={"num_neighbors": [5, 2]},
            force_reload=True,
        )

    def test_base_dataset(self, astrolab_dataset):
        """Test base dataset functionality."""
        print(f"Base dataset length: {len(astrolab_dataset)}")
        print(f"Base dataset sampler type: {type(astrolab_dataset.sampler)}")

        # Test that we can get a graph
        graph = astrolab_dataset.get(0)
        print(f"Base graph nodes: {graph.num_nodes}")
        print(f"Base graph edges: {graph.edge_index.shape[1]}")

        assert len(astrolab_dataset) > 0
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[1] > 0

    def test_knn_sampler(self, dataset_knn):
        """Test KNNSampler creates correct number of edges."""
        print(f"KNNSampler type: {type(dataset_knn.sampler)}")
        print(f"KNNSampler k value: {dataset_knn.sampler.k}")

        graph = dataset_knn.get(0)
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.edge_index.shape[1]}")
        print(f"Expected edges (k=2): {graph.num_nodes * 2}")

        # Verify sampler type
        assert type(dataset_knn.sampler).__name__ == "KNNSampler"
        assert dataset_knn.sampler.k == 2

        # Verify graph properties
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[1] > 0
        assert graph.edge_index.shape[0] == 2

        # For k=2, we should have approximately 2*num_nodes edges
        # (allowing for some variation due to graph structure)
        expected_edges = graph.num_nodes * 2
        actual_edges = graph.edge_index.shape[1]
        edge_ratio = actual_edges / expected_edges

        print(f"Edge ratio: {edge_ratio:.2f}")
        assert 0.5 <= edge_ratio <= 2.0, (
            f"Edge ratio {edge_ratio} outside expected range"
        )

    def test_radius_sampler(self, dataset_radius):
        """Test RadiusSampler creates reasonable number of edges."""
        print(f"RadiusSampler type: {type(dataset_radius.sampler)}")
        print(f"RadiusSampler radius: {dataset_radius.sampler.radius}")
        print(
            f"RadiusSampler max_num_neighbors: {dataset_radius.sampler.max_num_neighbors}"
        )

        graph = dataset_radius.get(0)
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.edge_index.shape[1]}")

        # Verify sampler type
        assert type(dataset_radius.sampler).__name__ == "RadiusSampler"
        assert dataset_radius.sampler.radius == 1.0
        assert dataset_radius.sampler.max_num_neighbors == 10

        # Verify graph properties
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[1] > 0
        assert graph.edge_index.shape[0] == 2

        # For radius sampling, edges should be reasonable
        max_possible_edges = graph.num_nodes * dataset_radius.sampler.max_num_neighbors
        actual_edges = graph.edge_index.shape[1]

        print(f"Max possible edges: {max_possible_edges}")
        print(f"Actual edges: {actual_edges}")
        assert actual_edges <= max_possible_edges

    def test_adaptive_radius_sampler(self, dataset_adaptive):
        """Test AdaptiveRadiusSampler creates reasonable number of edges."""
        print(f"AdaptiveRadiusSampler type: {type(dataset_adaptive.sampler)}")
        print(
            f"AdaptiveRadiusSampler target_neighbors: {dataset_adaptive.sampler.target_neighbors}"
        )

        graph = dataset_adaptive.get(0)
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.edge_index.shape[1]}")

        # Verify sampler type
        assert type(dataset_adaptive.sampler).__name__ == "AdaptiveRadiusSampler"
        assert dataset_adaptive.sampler.target_neighbors == 5

        # Verify graph properties
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[1] > 0
        assert graph.edge_index.shape[0] == 2

    def test_neighbor_subgraph_sampler(self, dataset_neighbor):
        """Test NeighborSubgraphSampler creates reasonable number of edges."""
        print(f"NeighborSubgraphSampler type: {type(dataset_neighbor.sampler)}")
        print(
            f"NeighborSubgraphSampler num_neighbors: {dataset_neighbor.sampler.num_neighbors}"
        )

        graph = dataset_neighbor.get(0)
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.edge_index.shape[1]}")

        # Verify sampler type
        assert type(dataset_neighbor.sampler).__name__ == "NeighborSubgraphSampler"
        assert dataset_neighbor.sampler.num_neighbors == [5, 2]

        # Verify graph properties
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[1] > 0
        assert graph.edge_index.shape[0] == 2

        # For neighbor sampling, edges should be based on num_neighbors[0]
        expected_edges = graph.num_nodes * dataset_neighbor.sampler.num_neighbors[0]
        actual_edges = graph.edge_index.shape[1]

        print(
            f"Expected edges (num_neighbors[0]={dataset_neighbor.sampler.num_neighbors[0]}): {expected_edges}"
        )
        print(f"Actual edges: {actual_edges}")

        # Allow some variation
        edge_ratio = actual_edges / expected_edges
        print(f"Edge ratio: {edge_ratio:.2f}")
        assert 0.5 <= edge_ratio <= 2.0, (
            f"Edge ratio {edge_ratio} outside expected range"
        )

    def test_sampler_edge_density_comparison(self):
        """Compare edge density across different samplers."""
        samplers = {
            "knn_k2": ("knn", {"k": 2}),
            "knn_k5": ("knn", {"k": 5}),
            "radius_small": ("radius", {"radius": 0.5, "max_num_neighbors": 5}),
            "radius_large": ("radius", {"radius": 2.0, "max_num_neighbors": 20}),
            "neighbor_sparse": ("neighbor", {"num_neighbors": [3, 1]}),
            "neighbor_dense": ("neighbor", {"num_neighbors": [10, 5]}),
        }

        results = {}

        for name, (strategy, kwargs) in samplers.items():
            print(f"\nTesting {name}...")
            dataset = AstroLabInMemoryDataset(
                survey_name="gaia",
                task="node_classification",
                sampling_strategy=strategy,
                sampler_kwargs=kwargs,
                force_reload=True,
            )

            graph = dataset.get(0)
            num_nodes = graph.num_nodes
            num_edges = graph.edge_index.shape[1]
            density = num_edges / (num_nodes * (num_nodes - 1))

            results[name] = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "avg_degree": num_edges / num_nodes,
            }

            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {num_edges}")
            print(f"  Density: {density:.6f}")
            print(f"  Avg degree: {num_edges / num_nodes:.2f}")

        # Verify that different samplers produce different densities
        densities = [results[name]["density"] for name in results]
        assert len(set(densities)) > 1, "All samplers produced same density"

        # Verify that k=2 produces fewer edges than k=5
        assert results["knn_k2"]["num_edges"] < results["knn_k5"]["num_edges"]

        # Verify that sparse neighbor sampling produces fewer edges than dense
        assert (
            results["neighbor_sparse"]["num_edges"]
            < results["neighbor_dense"]["num_edges"]
        )

    def test_sampler_memory_usage(self):
        """Test that streaming dataset uses reasonable memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Initial memory: {initial_memory:.2f} MB")

        # Create dataset and load a few graphs
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="knn",
            sampler_kwargs={"k": 2},
            force_reload=True,
        )

        memory_after_dataset = process.memory_info().rss / 1024 / 1024
        print(f"Memory after dataset creation: {memory_after_dataset:.2f} MB")

        # Load several graphs
        graphs = []
        for i in range(5):
            graph = dataset.get(i)
            graphs.append(graph)

        memory_after_graphs = process.memory_info().rss / 1024 / 1024
        print(f"Memory after loading 5 graphs: {memory_after_graphs:.2f} MB")

        memory_increase = memory_after_graphs - initial_memory
        print(f"Total memory increase: {memory_increase:.2f} MB")

        # Memory usage should be reasonable (< 1GB for 5 graphs)
        assert memory_increase < 1024, (
            f"Memory usage too high: {memory_increase:.2f} MB"
        )

        # Verify graphs are loaded correctly
        assert len(graphs) == 5
        for graph in graphs:
            assert graph.num_nodes > 0
            assert graph.edge_index.shape[1] > 0


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestSamplers()

    print("Testing KNNSampler...")
    dataset_knn = test_instance.dataset_knn()
    test_instance.test_knn_sampler(dataset_knn)

    print("\nTesting RadiusSampler...")
    dataset_radius = test_instance.dataset_radius()
    test_instance.test_radius_sampler(dataset_radius)

    print("\nTesting AdaptiveRadiusSampler...")
    dataset_adaptive = test_instance.dataset_adaptive()
    test_instance.test_adaptive_radius_sampler(dataset_adaptive)

    print("\nTesting NeighborSubgraphSampler...")
    dataset_neighbor = test_instance.dataset_neighbor()
    test_instance.test_neighbor_subgraph_sampler(dataset_neighbor)

    print("\nTesting sampler comparison...")
    test_instance.test_sampler_edge_density_comparison()

    print("\nTesting memory usage...")
    test_instance.test_sampler_memory_usage()

    print("\nAll tests passed!")
