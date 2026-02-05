"""
MemoryEfficientMixin for AstroLab Training
=========================================

Provides memory-efficient utilities for training (e.g., memmap, gradient checkpointing).
"""

from typing import Optional

import numpy as np
import torch


class MemoryEfficientMixin:
    """Mixin for memory-efficient training utilities."""

    def to_memmap(self, path: str) -> None:
        """
        Save model parameters to a numpy memmap file for efficient storage.
        Args:
            path: File path for memmap
        """
        if not hasattr(self, "parameters") or not callable(getattr(self, "parameters")):
            raise NotImplementedError("to_memmap requires self.parameters() method.")
        params = [p.detach().cpu().numpy() for p in self.parameters()]
        arr = np.concatenate([p.flatten() for p in params])
        memmap = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        memmap[:] = arr[:]
        memmap.flush()

    def load_from_memmap(
        self, path: str, device: Optional[torch.device] = None
    ) -> None:
        """
        Load model parameters from a numpy memmap file.
        Args:
            path: File path for memmap
            device: Device to load parameters to
        """
        if not hasattr(self, "parameters") or not callable(getattr(self, "parameters")):
            raise NotImplementedError(
                "load_from_memmap requires self.parameters() method."
            )
        arr = np.memmap(path, mode="r")
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(
                torch.from_numpy(arr[offset : offset + numel].reshape(p.shape))
            )
            offset += numel
        if device is not None:
            if not hasattr(self, "to") or not callable(getattr(self, "to")):
                raise NotImplementedError(
                    "load_from_memmap requires self.to() method for device transfer."
                )
            self.to(device)

    def gradient_checkpointing_enable(self) -> None:
        """
        Enable gradient checkpointing for memory efficiency.
        """
        try:
            pass

            self._gradient_checkpointing = True
        except ImportError:
            raise RuntimeError(
                "torch.utils.checkpoint is required for gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """
        Disable gradient checkpointing.
        """
        self._gradient_checkpointing = False
