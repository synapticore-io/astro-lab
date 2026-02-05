"""
Base Layer Components for AstroLab
==================================

Core building blocks for all astronomical neural network layers.
"""

# Set tensordict behavior globally for this module
import os

os.environ["LIST_TO_STACK"] = "1"

import tensordict

tensordict.set_list_to_stack(True)

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex
from torch_geometric.data import Data

# Import SurveyTensorDict for feature extraction
from astro_lab.tensors.survey import SurveyTensorDict

# ModernGraphEncoder, AdvancedTemporalEncoder, PointNetEncoder, TaskSpecificHead, GraphPooling aus components/base.py HIER EINFÜGEN


# --- ModernGraphEncoder ---
class ModernGraphEncoder(nn.Module): ...


# --- AdvancedTemporalEncoder ---
class AdvancedTemporalEncoder(nn.Module): ...


# --- PointNetEncoder ---
class PointNetEncoder(nn.Module): ...


# --- TaskSpecificHead ---
class TaskSpecificHead(nn.Module): ...


# --- GraphPooling ---
class GraphPooling(nn.Module): ...


class BaseLayer(nn.Module, ABC):
    """Base class for all AstroLab layers with SurveyTensorDict support."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the layer."""

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module != self:
                module.reset_parameters()


class FeatureExtractionLayer(BaseLayer):
    """
    Central feature extraction layer using SurveyTensorDict.

    This layer handles the conversion from SurveyTensorDict to features
    that can be used by graph layers, making the pipeline survey-agnostic.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        output_dim: Optional[int] = None,
        normalize_features: bool = True,
    ):
        super().__init__()

        self.feature_types = feature_types  # None = all available
        self.output_dim = output_dim
        self.normalize_features = normalize_features

        # Optional feature projection if output_dim is specified
        self.feature_proj = None
        if output_dim is not None:
            self.feature_proj = nn.Linear(0, output_dim)  # Will be set dynamically

    def forward(self, data: Union[Data, SurveyTensorDict, Tensor]) -> Tensor:
        """
        Extract features from various input types.

        Args:
            data: Can be SurveyTensorDict, PyG Data, or direct Tensor

        Returns:
            Feature tensor [N, F]
        """
        if isinstance(data, SurveyTensorDict):
            # Extract features from SurveyTensorDict
            features_dict = data.extract_features(self.feature_types)

            # Concatenate all tensor features
            feature_tensors = [
                v
                for v in features_dict.values()
                if isinstance(v, torch.Tensor) and v.dim() > 0
            ]

            if feature_tensors:
                features = torch.cat(feature_tensors, dim=1)
            else:
                # Fallback: use coordinates if available
                coords = data.get_coordinates()
                if coords is not None:
                    features = coords
                else:
                    raise ValueError("No features found in SurveyTensorDict")

        elif isinstance(data, Data):
            # PyG Data - use x if available, otherwise extract from pos
            if hasattr(data, "x") and data.x is not None:
                features = data.x
            elif hasattr(data, "pos") and data.pos is not None:
                features = data.pos
            else:
                raise ValueError("PyG Data has no features (x) or positions (pos)")

        elif isinstance(data, torch.Tensor):
            # Direct tensor input
            features = data

        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # Ensure 2D
        if features.dim() == 1:
            features = features.unsqueeze(1)

        # Normalize if requested
        if self.normalize_features:
            features = torch.nn.functional.normalize(features, dim=1)

        # Project to output dimension if specified
        if self.feature_proj is not None:
            if self.feature_proj.in_features == 0:
                # Initialize projection layer dynamically
                self.feature_proj = nn.Linear(
                    features.shape[1], self.output_dim or features.shape[1]
                ).to(features.device)
            features = self.feature_proj(features)

        return features


class BaseGraphLayer(BaseLayer):
    """Abstract base class for all graph layers with SurveyTensorDict support."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass through the layer."""

    def cache_edge_index(
        self, edge_index: Union[Tensor, EdgeIndex]
    ) -> Union[Tensor, EdgeIndex]:
        """Pass through edge_index without caching (torch.compile compatible)."""
        # 2025 Best Practice: No manual EdgeIndex wrapping for torch.compile compatibility
        return edge_index


class BasePoolingLayer(BaseLayer):
    """Abstract base class for pooling layers with task awareness."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        task: str = "node_classification",
        **kwargs,
    ) -> Tensor:
        """Pool node features to graph-level representation."""


class BaseAttentionLayer(BaseLayer):
    """Abstract base class for attention mechanisms."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @abstractmethod
    def compute_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute attention weights and optionally apply to values."""


class TensorDictLayer(BaseLayer):
    """Base class for layers that work with TensorDict."""

    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        pass_through: bool = True,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.pass_through = pass_through

    @abstractmethod
    def process_tensordict(self, td: TensorDict) -> Dict[str, Tensor]:
        """Process input tensordict and return output dict."""

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass with TensorDict."""
        # Extract inputs
        {key: td[key] for key in self.in_keys if key in td}

        # Process
        outputs = self.process_tensordict(td)

        # Update tensordict
        if self.pass_through:
            td = td.clone()
        else:
            td = TensorDict({}, batch_size=td.batch_size)

        for key, value in outputs.items():
            td[key] = value

        return td
