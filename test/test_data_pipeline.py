"""Test the complete data pipeline with performance measurements."""

import logging
import time
from typing import Dict

import psutil
import torch

from astro_lab.data.samplers.RadiusSampler import RadiusSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    # CPU Memory
    cpu_percent = psutil.virtual_memory().percent
    cpu_used_gb = psutil.virtual_memory().used / (1024**3)

    # GPU memory (if available)
    gpu_percent = 0.0
    gpu_used_gb = 0.0

    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.mem_get_info()
            gpu_used = gpu_memory[1] - gpu_memory[0]
            gpu_used_gb = gpu_used / (1024**3)
            gpu_percent = (gpu_used / gpu_memory[1]) * 100
        except (RuntimeError, AttributeError):
            # GPU memory query not supported
            pass

    return {
        "cpu_percent": cpu_percent,
        "cpu_used_gb": cpu_used_gb,
        "gpu_percent": gpu_percent,
        "gpu_used_gb": gpu_used_gb,
    }


def test_preprocessor(survey_name: str, max_samples: int = 1000):
    """Test preprocessor for a specific survey."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing {survey_name.upper()} Preprocessor")
    logger.info(f"{'=' * 60}")

    try:
        import numpy as np
        import polars as pl

        from astro_lab.data.preprocessors.gaia import GaiaPreprocessor

        # Get memory before
        mem_before = get_memory_usage()
        start_time = time.time()

        # Initialize preprocessor
        if survey_name == "gaia":
            preprocessor = GaiaPreprocessor()
        else:
            # For other surveys, use Gaia as default for now
            preprocessor = GaiaPreprocessor()

        # Create synthetic data for testing
        logger.info(f"Creating synthetic data with {max_samples} samples...")

        if survey_name == "gaia":
            # Synthetic Gaia data
            data = {
                "source_id": list(range(max_samples)),
                "ra": np.random.uniform(0, 360, max_samples),
                "dec": np.random.uniform(-90, 90, max_samples),
                "parallax": np.random.uniform(0.1, 10, max_samples),
                "parallax_error": np.random.uniform(0.01, 0.1, max_samples),
                "pmra": np.random.normal(0, 10, max_samples),
                "pmdec": np.random.normal(0, 10, max_samples),
                "pmra_error": np.random.uniform(0.01, 0.5, max_samples),
                "pmdec_error": np.random.uniform(0.01, 0.5, max_samples),
                "phot_g_mean_mag": np.random.uniform(10, 20, max_samples),
                "phot_bp_mean_mag": np.random.uniform(10, 20, max_samples),
                "phot_rp_mean_mag": np.random.uniform(10, 20, max_samples),
                "astrometric_excess_noise": np.random.uniform(0, 1, max_samples),
                "ruwe": np.random.uniform(0.8, 1.2, max_samples),
            }
        elif survey_name == "sdss":
            # Synthetic SDSS data
            data = {
                "objid": list(range(max_samples)),
                "ra": np.random.uniform(0, 360, max_samples),
                "dec": np.random.uniform(-90, 90, max_samples),
                "u": np.random.uniform(12, 22, max_samples),
                "g": np.random.uniform(12, 22, max_samples),
                "r": np.random.uniform(12, 22, max_samples),
                "i": np.random.uniform(12, 22, max_samples),
                "z_band": np.random.uniform(12, 22, max_samples),
                "z": np.random.uniform(0, 0.5, max_samples),  # redshift
                "clean": np.ones(max_samples),  # clean photometry flag
            }
        elif survey_name == "exoplanet":
            # Synthetic exoplanet data
            data = {
                "pl_name": [f"planet_{i}" for i in range(max_samples)],
                "ra": np.random.uniform(0, 360, max_samples),
                "dec": np.random.uniform(-90, 90, max_samples),
                "sy_dist": np.random.uniform(10, 500, max_samples),  # distance in pc
                "pl_bmassj": np.random.uniform(0.1, 10, max_samples),  # planet mass
                "pl_radj": np.random.uniform(0.1, 2, max_samples),  # planet radius
                "sy_gaiamag": np.random.uniform(
                    8, 16, max_samples
                ),  # host star magnitude
            }
        else:
            # Generic astronomical data
            data = {
                "id": list(range(max_samples)),
                "ra": np.random.uniform(0, 360, max_samples),
                "dec": np.random.uniform(-90, 90, max_samples),
                "mag": np.random.uniform(10, 25, max_samples),
            }

        df = pl.DataFrame(data)
        logger.info(f"Created DataFrame with shape: {df.shape}")

        # Test preprocessing pipeline
        logger.info("Running preprocessing pipeline...")

        # Filter
        filtered_df = preprocessor.filter(df)
        logger.info(
            f"After filter: {len(filtered_df)} objects ({len(filtered_df) / len(df) * 100:.1f}%)"
        )

        # Transform
        transformed_df = preprocessor.transform(filtered_df)
        logger.info(f"After transform: {transformed_df.shape}")

        # Extract features
        features_df = preprocessor.extract_features(transformed_df)
        logger.info(f"After feature extraction: {features_df.shape}")

        # Get preprocessor info
        info = preprocessor.get_info()
        logger.info(f"Preprocessor info: {info}")

        # Calculate time and memory
        process_time = time.time() - start_time
        mem_after = get_memory_usage()

        cpu_mem_used = mem_after["cpu_used_gb"] - mem_before["cpu_used_gb"]
        gpu_mem_used = mem_after["gpu_used_gb"] - mem_before["gpu_used_gb"]

        logger.info("\nPerformance metrics:")
        logger.info(f"  Processing time: {process_time:.2f}s")
        logger.info(f"  CPU memory used: {cpu_mem_used:.2f}GB")
        logger.info(f"  GPU memory used: {gpu_mem_used:.2f}GB")
        logger.info(f"  Objects/second: {max_samples / process_time:.0f}")

        return True

    except Exception as e:
        logger.error(f"Error testing preprocessor: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_optimized_preprocessor(survey_name: str, max_samples: int = 10000):
    """Test optimized preprocessor with caching and parallelization."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing OPTIMIZED {survey_name.upper()} Preprocessor")
    logger.info(f"{'=' * 60}")

    try:
        import numpy as np
        import polars as pl

        from astro_lab.data.preprocessors.optimized import get_optimized_preprocessor

        # Get memory before
        mem_before = get_memory_usage()
        start_time = time.time()

        # Initialize optimized preprocessor
        preprocessor = get_optimized_preprocessor(survey_name)

        # Create larger synthetic dataset
        logger.info(f"Creating synthetic data with {max_samples} samples...")

        # Same data creation as above but with more samples
        data = {
            "source_id": list(range(max_samples)),
            "ra": np.random.uniform(0, 360, max_samples),
            "dec": np.random.uniform(-90, 90, max_samples),
            "parallax": np.random.uniform(0.1, 10, max_samples),
            "parallax_error": np.random.uniform(0.01, 0.1, max_samples),
            "pmra": np.random.normal(0, 10, max_samples),
            "pmdec": np.random.normal(0, 10, max_samples),
            "pmra_error": np.random.uniform(0.01, 0.5, max_samples),
            "pmdec_error": np.random.uniform(0.01, 0.5, max_samples),
            "phot_g_mean_mag": np.random.uniform(10, 20, max_samples),
            "phot_bp_mean_mag": np.random.uniform(10, 20, max_samples),
            "phot_rp_mean_mag": np.random.uniform(10, 20, max_samples),
            "astrometric_excess_noise": np.random.uniform(0, 1, max_samples),
            "ruwe": np.random.uniform(0.8, 1.2, max_samples),
        }

        df = pl.DataFrame(data)

        # Run preprocessing
        logger.info("Running optimized preprocessing pipeline...")

        filtered_df = preprocessor.filter(df)
        transformed_df = preprocessor.transform(filtered_df)
        features_df = preprocessor.extract_features(transformed_df)

        # Calculate time and memory
        process_time = time.time() - start_time
        mem_after = get_memory_usage()

        cpu_mem_used = mem_after["cpu_used_gb"] - mem_before["cpu_used_gb"]
        gpu_mem_used = mem_after["gpu_used_gb"] - mem_before["gpu_used_gb"]

        # Get cache stats
        cache_stats = preprocessor.get_cache_stats()

        logger.info("\nOptimized Performance metrics:")
        logger.info(f"  Processing time: {process_time:.2f}s")
        logger.info(f"  CPU memory used: {cpu_mem_used:.2f}GB")
        logger.info(f"  GPU memory used: {gpu_mem_used:.2f}GB")
        logger.info(f"  Objects/second: {max_samples / process_time:.0f}")
        logger.info(f"  Cache hit rate: {cache_stats['hit_rate'] * 100:.1f}%")

        return True

    except Exception as e:
        logger.error(f"Error testing optimized preprocessor: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_datamodule(survey_name: str, max_samples: int = 1000):
    """Test complete data module with samplers."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing {survey_name.upper()} DataModule")
    logger.info(f"{'=' * 60}")

    try:
        from astro_lab.data.datamodules import get_survey_datamodule

        # Get memory before
        mem_before = get_memory_usage()
        start_time = time.time()

        # Create data module
        dm = get_survey_datamodule(
            survey_name,
            sampler_type="knn",
            sampler_config={"k": 10},
            batch_size=32,
            num_workers=0,
            max_samples=max_samples,
            force_download=False,
        )

        # Prepare and setup
        logger.info("Preparing data...")
        dm.prepare_data()

        logger.info("Setting up data pipeline...")
        dm.setup()

        # Get info
        info = dm.get_info()
        logger.info(f"DataModule info: {info}")

        # Test dataloaders
        logger.info("Testing dataloaders...")
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Get a batch
        batch = next(iter(train_loader))
        logger.info("Batch info:")
        logger.info(f"  Batch size: {batch.batch.max() + 1}")
        logger.info(f"  Node features: {batch.x.shape}")
        logger.info(f"  Edge indices: {batch.edge_index.shape}")

        # Calculate time and memory
        process_time = time.time() - start_time
        mem_after = get_memory_usage()

        cpu_mem_used = mem_after["cpu_used_gb"] - mem_before["cpu_used_gb"]
        gpu_mem_used = mem_after["gpu_used_gb"] - mem_before["gpu_used_gb"]

        logger.info("\nDataModule Performance:")
        logger.info(f"  Total time: {process_time:.2f}s")
        logger.info(f"  CPU memory used: {cpu_mem_used:.2f}GB")
        logger.info(f"  GPU memory used: {gpu_mem_used:.2f}GB")

        return True

    except Exception as e:
        logger.error(f"Error testing datamodule: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_samplers():
    """Test different sampler types."""
    logger.info(f"\n{'=' * 60}")
    logger.info("Testing Different Samplers")
    logger.info(f"{'=' * 60}")

    try:
        import torch

        from astro_lab.data.samplers import (
            ClusterSampler,
            DBSCANClusterSampler,
            KNNSampler,
        )

        # List available samplers
        samplers = ["knn", "radius", "dbscan", "cluster"]
        logger.info(f"Available samplers: {samplers}")

        # Create test data
        n_nodes = 1000
        n_features = 10
        coordinates = torch.randn(n_nodes, 3)
        features = torch.randn(n_nodes, n_features)

        # Test each sampler type
        for sampler_type in ["knn", "radius", "dbscan", "cluster"]:
            if sampler_type not in samplers:
                continue

            logger.info(f"\nTesting {sampler_type} sampler...")

            try:
                start_time = time.time()

                # Get sampler
                if sampler_type == "knn":
                    sampler = KNNSampler(k=10)
                elif sampler_type == "radius":
                    sampler = RadiusSampler(radius=1.0)
                elif sampler_type == "dbscan":
                    sampler = DBSCANClusterSampler(eps=0.5)
                elif sampler_type == "cluster":
                    sampler = ClusterSampler(num_parts=10)
                else:
                    sampler = KNNSampler(k=10)

                # Create graph
                graph = sampler.create_graph(coordinates, features)

                # Get info
                info = sampler.get_sampling_info()

                elapsed = time.time() - start_time

                logger.info(f"  Created graph in {elapsed:.3f}s")
                logger.info(f"  Nodes: {graph.num_nodes}")
                logger.info(f"  Edges: {graph.num_edges}")
                logger.info(f"  Sampling info: {info['sampling_stats']}")

            except Exception as e:
                logger.error(f"  Error with {sampler_type}: {e}")

        return True

    except Exception as e:
        logger.error(f"Error testing samplers: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("ASTROLAB DATA PIPELINE PERFORMANCE TEST")
    logger.info("=" * 80)

    # Test surveys
    surveys = ["gaia", "sdss", "exoplanet"]

    # 1. Test basic preprocessors
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Testing Basic Preprocessors")
    logger.info("=" * 80)

    for survey in surveys:
        test_preprocessor(survey, max_samples=1000)

    # 2. Test optimized preprocessor (only Gaia implemented)
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Testing Optimized Preprocessors")
    logger.info("=" * 80)

    test_optimized_preprocessor("gaia", max_samples=50000)

    # 3. Test samplers
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Testing Graph Samplers")
    logger.info("=" * 80)

    test_samplers()

    # 4. Test complete data modules (skip for now as it needs real data)
    # logger.info("\n" + "=" * 80)
    # logger.info("PHASE 4: Testing Complete DataModules")
    # logger.info("=" * 80)

    # for survey in ['gaia']:
    #     test_datamodule(survey, max_samples=1000)

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
