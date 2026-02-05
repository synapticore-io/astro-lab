"""
BatchProcessingMixin for AstroLab Training
=========================================

Provides utilities for processing large batches and lazy stacking of predictions during training.
Handles real astronomical data batches efficiently without synthetic data generation.
"""

from typing import List, Optional, Union

import torch
from tensordict import TensorDict
from torch_geometric.data import Data


class BatchProcessingMixin:
    """Mixin for batch processing utilities in training context.

    Designed for handling real astronomical data batches efficiently
    with proper memory management and device transfers.
    """

    def process_large_batch(
        self,
        batch: Union[Data, TensorDict],
        chunk_size: int = 1000,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Process a large batch by splitting into chunks and aggregating results.

        This is useful for processing large astronomical surveys that may not
        fit in GPU memory all at once.

        Args:
            batch: Data or TensorDict containing real astronomical data
            chunk_size: Size of each processing chunk
            device: Optional device to move data to

        Returns:
            Aggregated result (Tensor or TensorDict)
        """
        if isinstance(batch, TensorDict):
            # Get the first tensor to determine batch size
            first_key = list(batch.keys())[0]
            data = batch[first_key]
            total_size = data.shape[0]
        elif isinstance(batch, Data):
            data = batch.x
            total_size = data.shape[0]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        results = []

        for i in range(0, total_size, chunk_size):
            end_idx = min(i + chunk_size, total_size)

            if isinstance(batch, TensorDict):
                # Create chunk from TensorDict
                chunk_dict = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                        chunk_dict[key] = value[i:end_idx]
                    else:
                        chunk_dict[key] = value
                chunk = TensorDict(chunk_dict, batch_size=end_idx - i)
            else:
                # Create chunk from Data object
                chunk = data[i:end_idx]

            if device is not None:
                chunk = chunk.to(device)

            # Process chunk - this should be overridden in the actual model
            # For now, we just return the input (identity operation)
            processed_chunk = self._process_chunk(chunk)
            results.append(processed_chunk)

        # Concatenate results
        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        elif isinstance(results[0], TensorDict):
            return self._concatenate_tensordicts(results)
        else:
            return torch.cat(results, dim=0)

    def _process_chunk(
        self, chunk: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Process a single chunk of data.

        This method should be overridden in subclasses to implement
        the actual processing logic (e.g., model forward pass).

        Args:
            chunk: Single chunk of astronomical data

        Returns:
            Processed chunk
        """
        # Default implementation: identity operation
        # Override this in subclasses for actual processing
        return chunk

    def _concatenate_tensordicts(self, tensordict_list: List[TensorDict]) -> TensorDict:
        """
        Concatenate a list of TensorDicts along the batch dimension.

        Args:
            tensordict_list: List of TensorDicts to concatenate

        Returns:
            Concatenated TensorDict
        """
        if not tensordict_list:
            raise ValueError("Cannot concatenate empty list of TensorDicts")

        # Get all keys from the first TensorDict
        keys = tensordict_list[0].keys()

        # Concatenate each key
        concatenated = {}
        for key in keys:
            tensors = []
            for td in tensordict_list:
                if key in td and isinstance(td[key], torch.Tensor):
                    tensors.append(td[key])

            if tensors:
                concatenated[key] = torch.cat(tensors, dim=0)
            else:
                # Keep non-tensor values from first TensorDict
                concatenated[key] = tensordict_list[0][key]

        # Calculate total batch size
        total_batch_size = sum(len(td) for td in tensordict_list)

        return TensorDict(concatenated, batch_size=total_batch_size)

    def lazy_stack_predictions(
        self,
        data_list: List[Union[Data, TensorDict]],
        device: Optional[torch.device] = None,
    ) -> TensorDict:
        """
        Lazily stack predictions from a list of Data or TensorDict objects.

        Useful for aggregating predictions from multiple astronomical observations
        or survey fields into a single structure.

        Args:
            data_list: List of Data or TensorDict objects containing real predictions
            device: Optional device to move data to

        Returns:
            Stacked TensorDict with combined predictions
        """
        if not data_list:
            raise ValueError("data_list must not be empty")

        if isinstance(data_list[0], TensorDict):
            return self._stack_tensordicts(data_list, device)
        elif isinstance(data_list[0], Data):
            return self._stack_pyg_data(data_list, device)
        else:
            raise TypeError(f"Unsupported data type in data_list: {type(data_list[0])}")

    def _stack_tensordicts(
        self, tensordict_list: List[TensorDict], device: Optional[torch.device] = None
    ) -> TensorDict:
        """Stack a list of TensorDicts."""
        stacked = {}

        # Get all unique keys
        all_keys = set()
        for td in tensordict_list:
            all_keys.update(td.keys())

        for key in all_keys:
            tensors = []
            for td in tensordict_list:
                if key in td and isinstance(td[key], torch.Tensor):
                    tensor = td[key].to(device) if device is not None else td[key]
                    tensors.append(tensor)

            if tensors:
                stacked[key] = torch.cat(tensors, dim=0)

        # Calculate total batch size
        if stacked:
            total_batch_size = next(iter(stacked.values())).shape[0]
        else:
            total_batch_size = len(tensordict_list)

        return TensorDict(stacked, batch_size=total_batch_size)

    def _stack_pyg_data(
        self, data_list: List[Data], device: Optional[torch.device] = None
    ) -> TensorDict:
        """Stack a list of PyTorch Geometric Data objects."""
        stacked = {}

        # Common attributes to stack
        attrs_to_stack = ["x", "edge_index", "edge_attr", "y", "pos", "batch"]

        for attr in attrs_to_stack:
            tensors = []
            for data in data_list:
                if hasattr(data, attr) and getattr(data, attr) is not None:
                    tensor = getattr(data, attr)
                    if device is not None:
                        tensor = tensor.to(device)
                    tensors.append(tensor)

            if tensors:
                if attr == "edge_index":
                    # Handle edge indices specially - need to offset node indices
                    offset = 0
                    offset_edge_indices = []
                    for i, tensor in enumerate(tensors):
                        offset_tensor = tensor + offset
                        offset_edge_indices.append(offset_tensor)
                        # Update offset for next graph
                        if hasattr(data_list[i], "x") and data_list[i].x is not None:
                            offset += data_list[i].x.shape[0]
                    stacked[attr] = torch.cat(offset_edge_indices, dim=1)
                else:
                    stacked[attr] = torch.cat(tensors, dim=0)

        # Calculate batch size
        batch_size = stacked.get("x", torch.tensor([len(data_list)])).shape[0]

        return TensorDict(stacked, batch_size=batch_size)

    def compute_batch_statistics(self, batch: Union[Data, TensorDict]) -> dict:
        """
        Compute statistics for a batch of astronomical data.

        Args:
            batch: Batch of astronomical observations

        Returns:
            Dictionary with batch statistics
        """
        stats = {}

        if isinstance(batch, TensorDict):
            stats["batch_type"] = "TensorDict"
            stats["batch_size"] = len(batch)
            stats["keys"] = list(batch.keys())

            # Compute statistics for each tensor
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    stats[f"{key}_shape"] = list(value.shape)
                    if value.dtype.is_floating_point:
                        stats[f"{key}_mean"] = value.mean().item()
                        stats[f"{key}_std"] = value.std().item()
                        stats[f"{key}_min"] = value.min().item()
                        stats[f"{key}_max"] = value.max().item()

        elif isinstance(batch, Data):
            stats["batch_type"] = "PyG_Data"
            stats["num_nodes"] = batch.num_nodes
            stats["num_edges"] = batch.num_edges

            if hasattr(batch, "x") and batch.x is not None:
                stats["node_features_shape"] = list(batch.x.shape)
                if batch.x.dtype.is_floating_point:
                    stats["node_features_mean"] = batch.x.mean().item()
                    stats["node_features_std"] = batch.x.std().item()

            if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                stats["edge_features_shape"] = list(batch.edge_attr.shape)

        return stats

    def validate_batch_consistency(self, batch: Union[Data, TensorDict]) -> bool:
        """
        Validate that a batch has consistent dimensions and valid data.

        Args:
            batch: Batch to validate

        Returns:
            True if batch is valid, False otherwise
        """
        try:
            if isinstance(batch, TensorDict):
                # Check that all tensors have consistent batch dimension
                batch_sizes = []
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                        batch_sizes.append(value.shape[0])

                if batch_sizes and len(set(batch_sizes)) > 1:
                    return False  # Inconsistent batch sizes

                # Check for NaN or infinite values
                for key, value in batch.items():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.dtype.is_floating_point
                    ):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            return False

            elif isinstance(batch, Data):
                # Validate PyG Data object
                if hasattr(batch, "x") and batch.x is not None:
                    if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                        return False

                if hasattr(batch, "edge_index") and batch.edge_index is not None:
                    # Check edge indices are valid
                    if batch.edge_index.min() < 0:
                        return False
                    if hasattr(batch, "x") and batch.x is not None:
                        if batch.edge_index.max() >= batch.x.shape[0]:
                            return False

            return True

        except Exception:
            return False
