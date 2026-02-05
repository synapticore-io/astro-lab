"""Test the complete training integration with config fixes."""

import logging
import sys

sys.path.insert(0, "src")

from astro_lab.cli.train import main as train_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Test 1: Basic training with auto model selection
print("\n" + "=" * 60)
print("Test 1: Basic training with Gaia data (auto model)")
print("=" * 60)

args = ["gaia", "--max-epochs", "2", "--batch-size", "16", "--verbose"]

try:
    # Simulate CLI arguments
    import argparse

    parser = argparse.Namespace(
        survey="gaia",
        task=None,  # Should default to node_classification
        model_type=None,  # Should use recommended model (GAT)
        max_epochs=2,
        batch_size=16,
        learning_rate=None,
        hidden_dim=None,
        num_layers=None,
        config=None,
        checkpoint=None,
        verbose=True,
    )

    result = train_main(parser)
    if result == 0:
        print("✅ Test 1 passed: Basic training successful")
    else:
        print("❌ Test 1 failed")
except Exception as e:
    print(f"❌ Test 1 failed with error: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Training with specific model type
print("\n" + "=" * 60)
print("Test 2: Training with specific model (GCN)")
print("=" * 60)

try:
    parser = argparse.Namespace(
        survey="sdss",
        task="graph_classification",
        model_type="gcn",
        max_epochs=1,
        batch_size=8,
        learning_rate=0.01,
        hidden_dim=64,
        num_layers=2,
        config=None,
        checkpoint=None,
        verbose=True,
    )

    result = train_main(parser)
    if result == 0:
        print("✅ Test 2 passed: Specific model training successful")
    else:
        print("❌ Test 2 failed")
except Exception as e:
    print(f"❌ Test 2 failed with error: {e}")

print("\n" + "=" * 60)
print("Integration tests completed!")
print("=" * 60)
