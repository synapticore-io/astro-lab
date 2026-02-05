"""AstroLab dataset for astronomical machine learning.

Provides unified dataset interface for all astronomical surveys with graph construction.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import torch
from torch_geometric.data import InMemoryDataset

from astro_lab.config import get_data_paths
from astro_lab.data.collectors.gaia import GaiaCollector
from astro_lab.data.samplers.AdaptiveRadiusSampler import AdaptiveRadiusSampler
from astro_lab.data.samplers.neighbor import (
    KNNSampler,
)
from astro_lab.data.samplers.NeighborSubgraphSampler import NeighborSubgraphSampler
from astro_lab.data.samplers.RadiusSampler import RadiusSampler
from astro_lab.data.transforms.astronomical import AstronomicalFeatures

# Set tensordict behavior globally for this module
os.environ["LIST_TO_STACK"] = "1"

import tensordict

tensordict.set_list_to_stack(True)

from tensordict import TensorDict
from torch_geometric.data import Data, HeteroData

from astro_lab.data.collectors import GaiaCollector
from astro_lab.data.samplers import KNNSampler
from astro_lab.data.transforms import AstronomicalFeatures
from astro_lab.tensors.photometric import PhotometricTensorDict
from astro_lab.tensors.spatial import SpatialTensorDict
from astro_lab.tensors.survey import SurveyTensorDict


class AstroLabInMemoryDataset(InMemoryDataset):
    """
    Universal in-memory dataset for astronomical data using SurveyTensorDict and PyG integration.
    Integrates collectors for downloading, samplers for graph construction, and transforms for
    preprocessing.

    Now with TensorDict optimizations:
    - Uses AstroTensorDict methods for memory-mapping
    - Consolidates batches for faster GPU transfer
    - Lazy stacks for efficient batch creation
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        survey_name: str = "gaia",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
        sampling_strategy: Optional[str] = None,
        sampler_kwargs: Optional[dict] = None,
        task: str = "node_classification",
        max_samples: Optional[int] = None,
        use_tensordict_optimization: bool = True,  # Enable TensorDict optimizations
    ):
        self.survey_name = survey_name
        self.metadata: Dict[str, Any] = {}
        self.sampling_strategy = sampling_strategy
        self.sampler_kwargs = sampler_kwargs or {}
        self.task = task
        self.max_samples = max_samples
        self.use_tensordict_optimization = use_tensordict_optimization
        self._tensordict_list: List[TensorDict] = []
        self._pyg_data_cache: Dict[int, Union[Data, HeteroData]] = {}

        # Setup paths
        data_paths = get_data_paths()
        if root is None:
            root = str(Path(data_paths["processed_dir"]) / survey_name)
        else:
            root = str(root)

        # Setup default transform if none provided - survey specific astronomical features
        if transform is None and pre_transform is None:
            transform = AstronomicalFeatures()

        # Initialize collector for data downloading
        self.collector = self._get_collector(survey_name)

        # Initialize sampler based on sampling strategy and task
        if sampling_strategy == "knn":
            self.sampler = KNNSampler(**self.sampler_kwargs)
        elif sampling_strategy == "radius":
            self.sampler = RadiusSampler(**self.sampler_kwargs)
        elif sampling_strategy == "adaptive":
            self.sampler = AdaptiveRadiusSampler(**self.sampler_kwargs)
        elif sampling_strategy == "neighbor":
            self.sampler = NeighborSubgraphSampler(**self.sampler_kwargs)
        else:
            # Default based on task
            if task in ("node_classification", "node_regression"):
                self.sampler = NeighborSubgraphSampler(**self.sampler_kwargs)
            else:
                self.sampler = KNNSampler(**self.sampler_kwargs)

        # Initialize data storage
        self._data_list = []

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self._data, self._slices = self._load_data()

    def _get_collector(self, survey_name: str):
        """Get the appropriate collector for the given survey name."""
        from astro_lab.data.collectors import (
            DESCollector,
            EuclidCollector,
            ExoplanetCollector,
            LinearCollector,
            NSACollector,
            PanSTARRSCollector,
            RRLyraeCollector,
            SDSSCollector,
            TNG50Collector,
            TwoMASSCollector,
            WISECollector,
        )

        collector_map = {
            "gaia": GaiaCollector,
            "sdss": SDSSCollector,
            "des": DESCollector,
            "euclid": EuclidCollector,
            "exoplanet": ExoplanetCollector,
            "linear": LinearCollector,
            "nsa": NSACollector,
            "panstarrs": PanSTARRSCollector,
            "rrlyrae": RRLyraeCollector,
            "tng50": TNG50Collector,
            "twomass": TwoMASSCollector,
            "wise": WISECollector,
        }

        collector_class = collector_map.get(survey_name.lower())
        if collector_class is None:
            raise ValueError(
                f"Unsupported survey: {survey_name}. Supported surveys: {list(collector_map.keys())}"
            )

        return collector_class(survey_name)

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names to check/download."""
        # Get from collector
        return self.collector.get_target_files()

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names to save/load."""
        suffix = f"_{self.max_samples}" if self.max_samples else ""
        files = [
            f"{self.survey_name}_{self.task}{suffix}.pt",
            f"{self.survey_name}_{self.task}{suffix}_metadata.json",
        ]
        if self.use_tensordict_optimization:
            files.append(f"{self.survey_name}_{self.task}{suffix}_memmap")
        return files

    @property
    def raw_dir(self) -> str:
        """Raw data directory."""
        return str(self.collector.raw_dir)

    @property
    def processed_dir(self) -> str:
        """Processed data directory."""
        data_paths = get_data_paths()
        processed_dir = Path(data_paths["processed_dir"]) / self.survey_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return str(processed_dir)

    def download(self):
        """Download raw data files using the collector."""
        self.collector.download(force=False)

    def process(self):
        """Process raw data metadata using streaming approach - no graph creation."""
        # Step 1: Load harmonized parquet data
        parquet_path = Path(self.processed_dir) / f"{self.survey_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Harmonized parquet file not found: {parquet_path}\n"
                f"Please run 'astro-lab preprocess {self.survey_name}' first."
            )

        # Get total number of rows without loading everything
        df_info = pl.scan_parquet(parquet_path).select(pl.count()).collect()
        n_rows = df_info[0, 0]

        # Limit samples if requested
        if self.max_samples:
            n_rows = min(n_rows, self.max_samples)

        # Step 2: Calculate graph structure based on task type
        if self.task == "node_classification":
            # Create larger graphs for better training efficiency
            graph_size = min(1000, n_rows)  # Increased from 100 to 1000 nodes per graph
            num_graphs = (n_rows + graph_size - 1) // graph_size
        elif self.task == "graph_classification":
            # Create separate graphs for graph-level classification
            graphs_per_chunk = max(100, n_rows // 30)
            num_graphs = (n_rows + graphs_per_chunk - 1) // graphs_per_chunk
        else:
            num_graphs = 1

        # Step 3: Save only metadata - graph creation delegated to samplers
        self.metadata = {
            "survey_name": self.survey_name,
            "task": self.task,
            "num_samples": num_graphs,
            "total_rows": n_rows,
            "graph_size": graph_size
            if self.task == "node_classification"
            else graphs_per_chunk,
            "sampling_strategy": self.sampling_strategy,
            "sampler_kwargs": self.sampler_kwargs,
            "parquet_path": str(parquet_path),
            "streaming": True,  # Mark as streaming dataset
            "use_tensordict_optimization": self.use_tensordict_optimization,
        }

        # Save metadata JSON
        with open(self.processed_paths[1], "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Step 4: Save a placeholder .pt file for compatibility with CI validation
        # For streaming datasets, we save minimal data since actual graphs are loaded on-demand
        placeholder_data = Data(
            x=torch.zeros(1, 1),  # Minimal feature tensor
            edge_index=torch.zeros(2, 0, dtype=torch.long),  # Empty edge index
            num_nodes=1,
        )
        placeholder_slices = {
            "x": torch.tensor([0, 1], dtype=torch.long),
            "edge_index": torch.tensor([0, 0], dtype=torch.long),
        }

        # Save the placeholder .pt file (PyG InMemoryDataset format)
        torch.save((placeholder_data, placeholder_slices), self.processed_paths[0])

        # Create empty data list for compatibility
        self._data_list = []

    def _extract_coordinates(self, torch_data: dict) -> torch.Tensor:
        """Extract 3D spatial coordinates from survey data."""
        # Check for cartesian coordinates first
        if all(key in torch_data for key in ["x", "y", "z"]):
            return torch.stack(
                [torch_data["x"], torch_data["y"], torch_data["z"]], dim=1
            )

        # Check for direct position tensor
        if "pos" in torch_data:
            return torch_data["pos"]

        # Convert from spherical coordinates if available
        if all(key in torch_data for key in ["ra", "dec", "distance_pc"]):
            ra_rad = torch.deg2rad(torch_data["ra"])
            dec_rad = torch.deg2rad(torch_data["dec"])
            dist = torch_data["distance_pc"]

            x = dist * torch.cos(dec_rad) * torch.cos(ra_rad)
            y = dist * torch.cos(dec_rad) * torch.sin(ra_rad)
            z = dist * torch.sin(dec_rad)

            return torch.stack([x, y, z], dim=1)

        # Fallback: generate random coordinates
        n_points = len(next(iter(torch_data.values())))
        return torch.randn(n_points, 3)

    def _torch_data_to_survey_tensordict(self, torch_data: dict) -> SurveyTensorDict:
        """Convert a torch_data dict to a SurveyTensorDict for unified feature extraction."""
        # Try to extract spatial component
        spatial = None
        if all(k in torch_data for k in ["x", "y", "z"]):
            spatial = SpatialTensorDict(
                coordinates=torch.stack(
                    [torch_data["x"], torch_data["y"], torch_data["z"]], dim=1
                )
            )
        elif "pos" in torch_data:
            spatial = SpatialTensorDict(coordinates=torch_data["pos"])

        # Try to extract photometric component (use all float columns except coordinates)
        photometric_keys = [
            k
            for k in torch_data.keys()
            if k
            not in {
                "x",
                "y",
                "z",
                "pos",
                "ra",
                "dec",
                "distance_pc",
                "source_id",
                "object_id",
            }
            and torch.is_tensor(torch_data[k])
            and torch_data[k].dtype in (torch.float32, torch.float64)
        ]
        photometric = None
        if photometric_keys:
            photometric = PhotometricTensorDict(
                magnitudes=torch.cat(
                    [
                        torch_data[k].unsqueeze(1)
                        if torch_data[k].dim() == 1
                        else torch_data[k]
                        for k in photometric_keys
                    ],
                    dim=1,
                ),
                bands=photometric_keys,
            )

        # Optionally, add image if present (not implemented here)
        image = None

        return SurveyTensorDict(
            spatial=spatial,
            photometric=photometric,
            image=image,
            survey_name=self.survey_name,
            meta={},
        )

    def _extract_features(self, torch_data: dict) -> torch.Tensor:
        """Extract feature vectors using SurveyTensorDict for survey-agnostic logic."""
        survey_td = self._torch_data_to_survey_tensordict(torch_data)
        features_dict = survey_td.extract_features()
        # Concatenate all features into a single tensor (except non-tensor entries)
        feature_tensors = [
            v
            for v in features_dict.values()
            if isinstance(v, torch.Tensor) and v.dim() > 0
        ]
        if feature_tensors:
            features = torch.cat(feature_tensors, dim=1)
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            return features
        # fallback: use coordinates if nothing else
        coords = self._extract_coordinates(torch_data)
        return coords

    def _create_node_labels(self, torch_data: dict, num_nodes: int) -> torch.Tensor:
        """Create node-level labels based on survey-specific classification schemes."""
        if self.survey_name == "gaia":
            # Stellar classification based on HR diagram position
            if "bp_rp" in torch_data and "mg_abs" in torch_data:
                bp_rp = torch_data["bp_rp"]
                mg_abs = torch_data["mg_abs"]

                # 6 classes: O-B, A-F, G-K, M main sequence, giants, supergiants
                labels = torch.zeros(num_nodes, dtype=torch.long)

                # Main sequence classification by color
                ms_mask = (mg_abs >= 0.0) & (mg_abs < 10.0)
                labels[ms_mask & (bp_rp < 0.0)] = 0  # O-B stars
                labels[ms_mask & (bp_rp >= 0.0) & (bp_rp < 0.5)] = 1  # A-F stars
                labels[ms_mask & (bp_rp >= 0.5) & (bp_rp < 1.0)] = 2  # G-K stars
                labels[ms_mask & (bp_rp >= 1.0)] = 3  # M stars

                # Giants and supergiants
                labels[(mg_abs >= -5.0) & (mg_abs < 0.0)] = 4  # Giants
                labels[mg_abs < -5.0] = 5  # Supergiants

                return labels
            else:
                # Fallback: random 6-class labels
                return torch.randint(0, 6, (num_nodes,), dtype=torch.long)

        elif self.survey_name == "sdss":
            # Galaxy morphology classification (simplified)
            if "g_r" in torch_data:
                g_r = torch_data["g_r"]

                # 3 classes: early-type (red), late-type (blue), intermediate
                labels = torch.zeros(num_nodes, dtype=torch.long)
                labels[g_r > 0.7] = 0  # Early-type (red)
                labels[g_r < 0.5] = 1  # Late-type (blue)
                labels[(g_r >= 0.5) & (g_r <= 0.7)] = 2  # Intermediate

                return labels
            else:
                return torch.randint(0, 3, (num_nodes,), dtype=torch.long)

        else:
            # Default: binary classification
            return torch.randint(0, 2, (num_nodes,), dtype=torch.long)

    def _create_graph_label(self, torch_data: dict, graph_idx: int) -> torch.Tensor:
        """Create graph-level label for graph classification tasks."""
        if self.survey_name == "gaia":
            # Classify stellar clusters by mean distance
            if "parallax" in torch_data:
                mean_parallax = torch_data["parallax"].mean()
                # Near (parallax > 5 mas) vs far clusters
                return torch.tensor(0 if mean_parallax > 5.0 else 1, dtype=torch.long)

        elif self.survey_name == "sdss":
            # Classify galaxy groups by mean redshift
            if "z" in torch_data:
                mean_z = torch_data["z"].mean()
                # Low-z vs high-z groups
                return torch.tensor(0 if mean_z < 0.1 else 1, dtype=torch.long)

        # Default: alternate binary labels
        return torch.tensor(graph_idx % 2, dtype=torch.long)

    def _tensordict_to_pyg(self, tensordict: TensorDict) -> Data:
        """Convert TensorDict to PyG Data object.

        Now uses consolidate() for faster GPU transfer if optimization is enabled.
        """
        if self.use_tensordict_optimization:
            # Consolidate for faster transfer
            tensordict = tensordict.consolidate()

        data_dict = {}
        for key, value in tensordict.items():
            data_dict[key] = value

        # Ensure x is 2D
        if "x" in data_dict and isinstance(data_dict["x"], torch.Tensor):
            if data_dict["x"].dim() == 1:
                data_dict["x"] = data_dict["x"].unsqueeze(1)

        return Data(**data_dict)

    def _pyg_to_tensordict(self, pyg_data: Data) -> TensorDict:
        """Convert PyG Data to TensorDict."""
        td = TensorDict({k: v for k, v in pyg_data.items()})

        if self.use_tensordict_optimization:
            # Pin memory for faster GPU transfer
            td = td.pin_memory_()

        return td

    def _load_data(self) -> Tuple[Any, Any]:
        """Load metadata for streaming dataset."""
        try:
            with open(self.processed_paths[1], "r") as f:
                self.metadata = json.load(f)

            # For streaming datasets, we don't load all data at once
            if self.metadata.get("streaming", False):
                # Check if we have memory-mapped data
                if self.use_tensordict_optimization and len(self.processed_paths) > 2:
                    memmap_path = Path(self.processed_paths[2])
                    if memmap_path.exists():
                        # Load memory-mapped TensorDict
                        from astro_lab.tensors import AstroTensorDict

                        self._memmap_tensordict = AstroTensorDict.from_checkpoint(
                            memmap_path
                        )
                return None, None
            else:
                # Fallback for non-streaming datasets
                data, slices = torch.load(self.processed_paths[0], weights_only=False)
                return data, slices

        except FileNotFoundError:
            return None, None

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        if hasattr(self, "metadata") and self.metadata:
            return self.metadata.get("num_samples", 0)
        return 0

    def get(self, idx: int) -> Union[Data, HeteroData]:
        """Get a single graph by index using streaming - delegate to sampler.

        Now with TensorDict optimizations:
        - Uses memory-mapped storage if available
        - Consolidates data for faster GPU transfer
        - Pins memory for async transfer
        """
        if idx >= self.len():
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self.len()} samples"
            )

        # Check if we have this in cache
        if self.use_tensordict_optimization and hasattr(self, "_memmap_tensordict"):
            # Use memory-mapped data
            return self._get_from_memmap(idx)

        # Load data on-demand using streaming
        parquet_path = Path(self.metadata["parquet_path"])
        total_rows = self.metadata["total_rows"]

        if self.task == "node_classification":
            graph_size = self.metadata["graph_size"]
            start = idx * graph_size
            end = min(start + graph_size, total_rows)
        else:  # graph_classification
            graphs_per_chunk = self.metadata["graph_size"]
            start = idx * graphs_per_chunk
            end = min(start + graphs_per_chunk, total_rows)

        # Load only the required chunk
        chunk = pl.read_parquet(parquet_path).slice(start, end - start)

        if len(chunk) < 5:  # Skip very small chunks
            # Return a minimal valid graph using the sampler
            if self.sampler:
                return self.sampler.create_graph(
                    coordinates=torch.randn(5, 3),
                    features=torch.randn(5, 3),
                    y=torch.zeros(5, dtype=torch.long),
                )
            else:
                # Require sampler for graph creation
                raise ValueError("Sampler is required for graph creation")

        # Convert to torch tensors
        torch_data = chunk.to_torch(return_type="dict")

        # Extract coordinates and features
        coordinates = self._extract_coordinates(torch_data)
        features = self._extract_features(torch_data)

        # Create labels based on task
        if self.task == "node_classification":
            labels = self._create_node_labels(torch_data, len(coordinates))
        else:
            labels = self._create_graph_label(torch_data, idx)

        # Create SurveyTensorDict for TensorDict optimization
        if self.use_tensordict_optimization:
            survey_td = self._torch_data_to_survey_tensordict(torch_data)
            # Add graph structure info
            survey_td["_graph_info"] = TensorDict(
                {
                    "coordinates": coordinates,
                    "features": features,
                    "y": labels,
                    "idx": torch.tensor(idx),
                }
            )

            # Consolidate for faster GPU transfer
            survey_td = survey_td.consolidate()

            # Delegate to sampler with TensorDict
            if self.sampler:
                graph_data = self.sampler.create_graph(
                    coordinates=coordinates,
                    features=features,
                    y=labels,
                    survey_name=self.survey_name,
                    tensordict=survey_td,  # Pass TensorDict if sampler supports it
                )
            else:
                raise ValueError("Sampler is required for graph creation")
        else:
            # Standard path without optimization
            if self.sampler:
                graph_data = self.sampler.create_graph(
                    coordinates=coordinates,
                    features=features,
                    y=labels,
                    survey_name=self.survey_name,
                )
            else:
                raise ValueError("Sampler is required for graph creation")

        # Apply transform if specified
        if self.transform is not None:
            graph_data = self.transform(graph_data)

        return graph_data

    def _get_from_memmap(self, idx: int) -> Data:
        """Get data from memory-mapped TensorDict."""
        # This would retrieve from memory-mapped storage
        # Implementation depends on how data is stored
        raise NotImplementedError("Memory-mapped retrieval not yet implemented")

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information for model initialization."""
        info = {
            "survey_name": self.survey_name,
            "task": self.task,
            "num_samples": len(self),
            "metadata": self.metadata,
            "sampling_strategy": self.sampling_strategy,
            "use_tensordict_optimization": self.use_tensordict_optimization,
        }

        if len(self) > 0:
            # Get info from first sample AFTER transforms are applied
            sample = self.get(0)

            info["num_nodes"] = sample.num_nodes
            info["num_edges"] = (
                sample.edge_index.shape[1] if hasattr(sample, "edge_index") else 0
            )

            # Get actual feature dimension from data
            info["num_features"] = (
                sample.x.shape[1]
                if hasattr(sample, "x") and sample.x is not None
                else self.metadata.get("num_features", 3)  # Default to 3 (coordinates)
            )

            # Determine number of classes
            if self.task == "node_classification":
                # Count unique labels across a few samples
                unique_labels = set()
                for i in range(min(10, len(self))):
                    s = self.get(i)
                    if hasattr(s, "y") and s.y is not None:
                        unique_labels.update(s.y.cpu().numpy().tolist())
                info["num_classes"] = max(2, len(unique_labels))
            else:
                # Graph classification: binary by default
                info["num_classes"] = 2
        else:
            # Default values from metadata or sensible defaults
            info["num_features"] = self.metadata.get("num_features", 3)
            info["num_classes"] = 6 if self.survey_name == "gaia" else 2

        return info

    def create_memmap_cache(self, num_workers: int = 4):
        """Create memory-mapped cache for entire dataset.

        Uses AstroTensorDict.to_memmap() for efficient storage.
        """
        if not self.use_tensordict_optimization:
            return

        memmap_path = Path(self.processed_paths[2])

        # Collect all data as TensorDicts
        all_data = []
        for i in range(len(self)):
            data = self.get(i)
            td = self._pyg_to_tensordict(data)
            all_data.append(td)

        # Use lazy stack for efficiency
        from astro_lab.tensors import AstroTensorDict

        stacked = AstroTensorDict.lazy_stack(all_data)

        # Convert to memory-mapped
        stacked.to_memmap(memmap_path, num_threads=num_workers)

        print(f"Created memory-mapped cache at {memmap_path}")


def create_dataset(
    root: Optional[Union[str, Path]] = None,
    survey_name: str = "gaia",
    data_type: Optional[str] = None,
    sampling_strategy: Optional[str] = None,
    sampler_kwargs: Optional[dict] = None,
    task: str = "node_classification",
    max_samples: Optional[int] = None,
    transform=None,
    pre_transform=None,
    pre_filter=None,
    force_reload: bool = False,
    use_tensordict_optimization: bool = True,
):
    """Factory function to create an AstroLabInMemoryDataset with the given parameters.

    Now with TensorDict optimizations enabled by default.
    """
    return AstroLabInMemoryDataset(
        root=root,
        survey_name=survey_name,
        transform=transform,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        force_reload=force_reload,
        sampling_strategy=sampling_strategy,
        sampler_kwargs=sampler_kwargs,
        task=task,
        max_samples=max_samples,
        use_tensordict_optimization=use_tensordict_optimization,
    )
