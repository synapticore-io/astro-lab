"""
Base TensorDict for AstroLab
============================

Base class for all astronomical TensorDicts leveraging native TensorDict features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from tensordict import MemoryMappedTensor, TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


class AstroTensorDict(TensorDict):
    """Base class for all astronomical TensorDicts.

    Leverages native TensorDict features:
    - Efficient batch operations
    - Memory-mapped storage
    - Lazy operations
    - Device management
    - torch.compile compatibility
    """

    def __init__(
        self,
        data: Dict[str, Any] = None,
        batch_size: Optional[torch.Size] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """Initialize AstroTensorDict.

        Args:
            data: Dictionary of tensors
            batch_size: Batch dimensions
            device: Device for tensors
            **kwargs: Additional TensorDict arguments
        """
        # Initialize with empty dict if no data
        if data is None:
            data = {}

        # Ensure batch_size if not provided
        if batch_size is None and data:
            # Infer from first tensor
            for v in data.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    batch_size = torch.Size([v.shape[0]])
                    break

        super().__init__(data, batch_size=batch_size, device=device, **kwargs)

        # Add astronomical metadata as sub-tensordict
        if "_metadata" not in self.keys():
            self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata sub-tensordict."""
        from datetime import datetime

        # Store metadata as nested TensorDict for consistency
        self["_metadata"] = TensorDict(
            {
                "tensor_type": self.__class__.__name__,
                "creation_time": datetime.now().isoformat(),
                "survey": getattr(self, "survey", "unknown"),
            },
            batch_size=torch.Size([]),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "AstroTensorDict":
        """Create from dictionary with automatic batching."""
        return cls(data, **kwargs)

    @classmethod
    def lazy_stack(
        cls, tensordicts: List["AstroTensorDict"], dim: int = 0
    ) -> "AstroTensorDict":
        """Lazy stack multiple AstroTensorDicts.

        Uses TensorDict's lazy stacking for memory efficiency.
        """
        # Use parent's lazy_stack
        stacked = TensorDict.lazy_stack(tensordicts, dim=dim)
        # Convert back to our type
        return cls(stacked.to_dict(), batch_size=stacked.batch_size)

    def to_memmap(
        self, path: Union[str, Path], num_threads: int = 16
    ) -> "AstroTensorDict":
        """Convert to memory-mapped storage for large datasets.

        Args:
            path: Directory to store memory-mapped files
            num_threads: Number of threads for parallel writing

        Returns:
            Memory-mapped AstroTensorDict
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Use native memmap functionality
        memmap_td = self.memmap_(str(path), num_threads=num_threads)

        # Log operation
        self._add_history("to_memmap", path=str(path))

        return memmap_td

    def consolidate(self, device: Optional[torch.device] = None) -> "AstroTensorDict":
        """Consolidate for fast inter-node communication.

        Creates a single contiguous tensor for all data,
        enabling fast serialization and transfer.
        """
        # Use native consolidate
        consolidated = super().consolidate()

        if device is not None:
            consolidated = consolidated.to(device)

        self._add_history("consolidate", device=str(device))
        return consolidated

    def share_memory_(self) -> "AstroTensorDict":
        """Enable shared memory for multiprocessing."""
        super().share_memory_()
        self._add_history("share_memory")
        return self

    def pin_memory_(self) -> "AstroTensorDict":
        """Pin memory for faster GPU transfer."""
        super().pin_memory_()
        self._add_history("pin_memory")
        return self

    @torch.compile(mode="reduce-overhead", dynamic=True)
    def apply_astronomical_transform(self, transform_fn: callable) -> "AstroTensorDict":
        """Apply transformation with torch.compile optimization.

        Args:
            transform_fn: Function to apply to each tensor

        Returns:
            Transformed TensorDict
        """
        return self.apply(transform_fn)

    def split_batch(self, split_size: int) -> List["AstroTensorDict"]:
        """Split into smaller batches for distributed processing."""
        n_splits = (self.batch_size[0] + split_size - 1) // split_size

        splits = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, self.batch_size[0])
            splits.append(self[start_idx:end_idx])

        return splits

    def merge_batch(self, others: List["AstroTensorDict"]) -> "AstroTensorDict":
        """Merge multiple batches efficiently."""
        all_tds = [self] + others
        return torch.cat(all_tds, dim=0)

    def to_module_input(self) -> TensorDict:
        """Convert to format expected by TensorDictModule."""
        # Remove metadata for module processing
        module_td = self.clone()
        if "_metadata" in module_td.keys():
            del module_td["_metadata"]
        return module_td

    def _add_history(self, operation: str, **details):
        """Add operation to history."""
        if "_metadata" not in self:
            self._init_metadata()

        if "history" not in self["_metadata"]:
            self["_metadata"]["history"] = []

        from datetime import datetime

        self["_metadata"]["history"].append(
            {"operation": operation, "timestamp": datetime.now().isoformat(), **details}
        )

    def extract_features(
        self, feature_types: Optional[List[str]] = None, as_dict: bool = False
    ) -> Union[TensorDict, Dict[str, torch.Tensor]]:
        """Extract features using native TensorDict operations.

        Args:
            feature_types: Types of features to extract
            as_dict: Return as dict instead of TensorDict

        Returns:
            Features as TensorDict or dict
        """
        # Create sub-tensordict with selected features
        if feature_types is None:
            # All features except metadata
            feature_td = self.select(*[k for k in self.keys() if k != "_metadata"])
        else:
            # Select by feature type
            selected_keys = []
            for key in self.keys():
                if (
                    key != "_metadata"
                    and self._classify_feature_type(key) in feature_types
                ):
                    selected_keys.append(key)
            feature_td = (
                self.select(*selected_keys) if selected_keys else TensorDict({})
            )

        return feature_td.to_dict() if as_dict else feature_td

    def _classify_feature_type(self, key: str) -> str:
        """Classify tensor key by feature type."""
        key_lower = key.lower()

        if any(k in key_lower for k in ["coord", "pos", "x", "y", "z", "ra", "dec"]):
            return "spatial"
        elif any(k in key_lower for k in ["time", "epoch", "mjd", "jd"]):
            return "temporal"
        elif any(k in key_lower for k in ["mag", "flux", "color", "g", "r", "i"]):
            return "photometric"
        elif any(k in key_lower for k in ["pm", "velocity", "parallax"]):
            return "kinematic"
        else:
            return "generic"

    @property
    def n_objects(self) -> int:
        """Number of astronomical objects."""
        return self.batch_size[0] if self.batch_size else 0

    def select_objects(self, indices: torch.Tensor) -> "AstroTensorDict":
        """Select subset of objects using native indexing."""
        return self[indices]

    def filter_by_condition(self, mask: torch.Tensor) -> "AstroTensorDict":
        """Filter using boolean mask."""
        return self[mask]

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = True
    ) -> "AstroTensorDict":
        """Async device transfer with TensorDict optimization."""
        return super().to(device, non_blocking=non_blocking)

    # === New Performance Methods ===

    def create_memmap_tensor(
        self,
        key: str,
        shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Create a memory-mapped tensor for large data.

        Args:
            key: Key for the tensor
            shape: Shape of the tensor
            dtype: Data type
            path: Optional path for the memory-mapped file
        """
        if path is None:
            # Use temporary directory
            import tempfile

            path = Path(tempfile.mkdtemp()) / f"{key}.memmap"
        else:
            path = Path(path)

        # Create memory-mapped tensor
        self[key] = MemoryMappedTensor.empty(
            shape=shape, dtype=dtype, filename=str(path)
        )

        self._add_history("create_memmap_tensor", key=key, shape=str(shape))

    def batch_compute(
        self,
        compute_fn: Callable,
        batch_size: int = 1000,
        output_key: str = "computed",
        input_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> "AstroTensorDict":
        """Compute function in batches to avoid OOM.

        Args:
            compute_fn: Function to apply to batches
            batch_size: Size of each batch
            output_key: Key to store results
            input_keys: Keys to pass to compute_fn (default: all non-metadata)
            **kwargs: Additional arguments for compute_fn

        Returns:
            Self with computed results
        """
        if input_keys is None:
            input_keys = [k for k in self.keys() if k != "_metadata"]

        n_objects = self.n_objects
        results = []

        # Process in batches
        for i in range(0, n_objects, batch_size):
            end_idx = min(i + batch_size, n_objects)

            # Extract batch
            batch_inputs = {}
            for key in input_keys:
                if key in self.keys():
                    batch_inputs[key] = self[key][i:end_idx]

            # Compute
            batch_result = compute_fn(batch_inputs, **kwargs)
            results.append(batch_result)

        # Concatenate results
        if results:
            if isinstance(results[0], torch.Tensor):
                self[output_key] = torch.cat(results, dim=0)
            elif isinstance(results[0], TensorDict):
                # Merge TensorDicts
                merged = TensorDict.cat(results, dim=0)
                for k, v in merged.items():
                    self[f"{output_key}_{k}"] = v

        self._add_history("batch_compute", output_key=output_key, batch_size=batch_size)
        return self

    def checkpoint(self, path: Union[str, Path]) -> None:
        """Save consolidated checkpoint for fast loading.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Consolidate first for faster save/load
        consolidated = self.consolidate()

        # Save with torch
        torch.save(consolidated.to_dict(), path)

        self._add_history("checkpoint", path=str(path))

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], device: Optional[torch.device] = None
    ) -> "AstroTensorDict":
        """Load from checkpoint with optional device placement.

        Args:
            path: Path to checkpoint
            device: Device to load to

        Returns:
            Loaded AstroTensorDict
        """
        data = torch.load(path, map_location=device or "cpu")
        return cls(data)

    def to_tensordict_module(
        self,
        module: nn.Module,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ) -> TensorDictModule:
        """Wrap a module for native TensorDict I/O.

        Args:
            module: PyTorch module to wrap
            in_keys: Input keys from TensorDict
            out_keys: Output keys to TensorDict

        Returns:
            TensorDictModule
        """
        if in_keys is None:
            in_keys = [k for k in self.keys() if k != "_metadata"]
        if out_keys is None:
            out_keys = ["output"]

        return TensorDictModule(module=module, in_keys=in_keys, out_keys=out_keys)
