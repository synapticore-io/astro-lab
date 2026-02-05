"""Manual script to process Gaia data with the fixed dataset."""

import shutil
from pathlib import Path

# First, clean up old processed data
processed_dir = Path("data/processed/gaia")
if processed_dir.exists():
    print(f"Removing old processed data from {processed_dir}")
    shutil.rmtree(processed_dir)

# Create fresh directory
processed_dir.mkdir(parents=True, exist_ok=True)
print(f"Created fresh directory: {processed_dir}")

# Now process the data using the dataset directly
from astro_lab.data import AstroLabInMemoryDataset

print("\nProcessing Gaia data...")
dataset = AstroLabInMemoryDataset(
    survey_name="gaia",
    sampling_strategy="knn",
    sampler_kwargs={"k": 8},
    force_reload=True,  # Force reprocessing
)

# Trigger processing
print("Calling dataset.process()...")
dataset.process(batch_size=10000, chunk_size=500)

# Check the results
info = dataset.get_info()
print(f"\nDataset info after processing:")
print(f"  Survey: {info['survey_name']}")
print(f"  Samples: {info['num_samples']}")
print(f"  Features: {info['num_features']}")
print(f"  Classes: {info['num_classes']}")
print(f"  Metadata: {info['metadata']}")

# Verify a sample
if len(dataset) > 0:
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  x shape: {sample.x.shape}")
    print(f"  y shape: {sample.y.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  Has train_mask: {hasattr(sample, 'train_mask')}")
    if hasattr(sample, "train_mask"):
        print(f"  Train mask sum: {sample.train_mask.sum()}")
        print(f"  Val mask sum: {sample.val_mask.sum()}")
        print(f"  Test mask sum: {sample.test_mask.sum()}")

print("\n✅ Data processing complete!")
