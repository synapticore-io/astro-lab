"""
Pooling Layers for AstroLab Models
=================================

Advanced pooling strategies for astronomical graph data.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import (
    ASAPooling,
    SAGPooling,
    TopKPooling,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter

from .base import BasePoolingLayer


class MultiScalePooling(BasePoolingLayer):
    """
    Multi-scale pooling for capturing features at different granularities.

    Particularly useful for astronomical data with hierarchical structure
    (stars -> clusters -> galaxies -> superclusters).
    """

    def __init__(
        self,
        in_channels: int,
        scales: List[int] = [1, 5, 10],
        pooling_method: str = "mean",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.scales = scales
        self.pooling_method = pooling_method

        # Create pooling layers for each scale
        self.scale_transforms = nn.ModuleList()
        for scale in scales:
            if scale > 1:
                # Coarsening transformation
                self.scale_transforms.append(nn.Linear(in_channels, in_channels))
            else:
                self.scale_transforms.append(nn.Identity())

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply multi-scale pooling."""

        pooled_features = []

        for scale, transform in zip(self.scales, self.scale_transforms):
            # Transform features
            x_transformed = transform(x)

            # Apply pooling at this scale
            if scale > 1 and edge_index is not None:
                # Coarsen graph by scale factor
                x_coarse = self._coarsen_features(x_transformed, edge_index, scale)
            else:
                x_coarse = x_transformed

            # Pool to graph level
            if self.pooling_method == "mean":
                pooled = (
                    global_mean_pool(x_coarse, batch)
                    if batch is not None
                    else x_coarse.mean(dim=0, keepdim=True)
                )
            elif self.pooling_method == "max":
                pooled = (
                    global_max_pool(x_coarse, batch)
                    if batch is not None
                    else x_coarse.max(dim=0)[0].unsqueeze(0)
                )
            elif self.pooling_method == "sum":
                pooled = (
                    global_add_pool(x_coarse, batch)
                    if batch is not None
                    else x_coarse.sum(dim=0, keepdim=True)
                )

            pooled_features.append(pooled)

        # Concatenate scale features
        return torch.cat(pooled_features, dim=-1)

    def _coarsen_features(self, x: Tensor, edge_index: Tensor, scale: int) -> Tensor:
        """Coarsen features by grouping nearby nodes."""
        # Simple implementation - in practice, use graph coarsening algorithms
        # For now, just apply strided sampling
        return x[::scale]


class AttentivePooling(BasePoolingLayer):
    """
    Attention-based pooling that learns to focus on important nodes.

    Useful for astronomical data where certain objects (e.g., bright stars,
    galaxy centers) are more informative than others.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.temperature = temperature

        # Multi-head attention for importance scoring
        self.attention = nn.MultiheadAttention(
            in_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query vector (what to look for)
        self.query = nn.Parameter(torch.randn(1, 1, in_channels))

        # Output projection
        self.output_proj = nn.Linear(in_channels, hidden_channels)

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply attentive pooling."""

        if batch is None:
            # Single graph
            return self._pool_single_graph(x, mask)
        else:
            # Batched graphs
            batch_size = batch.max().item() + 1
            pooled = []

            for i in range(batch_size):
                graph_mask = batch == i
                graph_nodes = x[graph_mask]

                # Create attention mask if needed
                if mask is not None:
                    graph_node_mask = mask[graph_mask]
                else:
                    graph_node_mask = None

                pooled_graph = self._pool_single_graph(graph_nodes, graph_node_mask)
                pooled.append(pooled_graph)

            return torch.cat(pooled, dim=0)

    def _pool_single_graph(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool a single graph using attention."""

        # Add batch dimension
        x_batched = x.unsqueeze(0)  # [1, N, D]

        # Expand query to match
        query = self.query.expand(1, -1, -1)  # [1, 1, D]

        # Apply attention
        attn_output, attn_weights = self.attention(
            query,
            x_batched,
            x_batched,
            key_padding_mask=mask.unsqueeze(0) if mask is not None else None,
        )

        # Remove batch dimension and project
        pooled = attn_output.squeeze(0)  # [1, D]
        return self.output_proj(pooled)


class HierarchicalPooling(BasePoolingLayer):
    """
    Hierarchical pooling using learnable clustering.

    Ideal for astronomical data with natural hierarchies
    (e.g., stellar systems, clusters, galaxies).
    """

    def __init__(
        self, in_channels: int, ratio: float = 0.5, method: str = "top_k", **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.method = method

        # Create appropriate pooling layer
        if method == "top_k":
            self.pool = TopKPooling(in_channels, ratio, **kwargs)
        elif method == "sag":
            self.pool = SAGPooling(in_channels, ratio, **kwargs)
        elif method == "asa":
            self.pool = ASAPooling(in_channels, ratio, **kwargs)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        """
        Apply hierarchical pooling.

        Returns:
            x: Pooled node features
            edge_index: Pooled edge indices
            batch: Pooled batch assignment
            edge_attr: Pooled edge attributes (if provided)
            perm: Node selection indices
            score: Node importance scores
        """

        return self.pool(x, edge_index, batch=batch, edge_attr=edge_attr)


class StatisticalPooling(BasePoolingLayer):
    """
    Pooling using statistical moments.

    Captures distribution properties of astronomical features
    (mean, variance, skewness, etc.).
    """

    def __init__(
        self,
        moments: List[str] = ["mean", "std", "min", "max"],
        dim: int = 0,
    ):
        super().__init__()

        self.moments = moments
        self.dim = dim

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute statistical moments."""

        pooled_features = []

        for moment in self.moments:
            if moment == "mean":
                pooled = (
                    global_mean_pool(x, batch)
                    if batch is not None
                    else x.mean(dim=self.dim, keepdim=True)
                )
            elif moment == "std":
                if batch is not None:
                    pooled = scatter(x, batch, dim=0, reduce="std")
                else:
                    pooled = x.std(dim=self.dim, keepdim=True)
            elif moment == "min":
                if batch is not None:
                    pooled = scatter(x, batch, dim=0, reduce="min")
                else:
                    pooled = x.min(dim=self.dim, keepdim=True)[0]
            elif moment == "max":
                if batch is not None:
                    pooled = scatter(x, batch, dim=0, reduce="max")
                else:
                    pooled = x.max(dim=self.dim, keepdim=True)[0]
            elif moment == "median":
                if batch is not None:
                    # Custom median pooling
                    batch_size = batch.max().item() + 1
                    median_pooled = []
                    for i in range(batch_size):
                        mask = batch == i
                        if mask.any():
                            median_pooled.append(x[mask].median(dim=0, keepdim=True)[0])
                        else:
                            median_pooled.append(
                                torch.zeros(1, x.size(-1), device=x.device)
                            )
                    pooled = torch.cat(median_pooled, dim=0)
                else:
                    pooled = x.median(dim=self.dim, keepdim=True)[0]
            else:
                continue

            pooled_features.append(pooled)

        return torch.cat(pooled_features, dim=-1)


class AdaptivePooling(BasePoolingLayer):
    """
    Adaptive pooling that learns which pooling method to use.

    Different astronomical surveys or object types may benefit
    from different pooling strategies.
    """

    def __init__(
        self,
        in_channels: int,
        pooling_methods: List[str] = ["mean", "max", "attention"],
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.pooling_methods = pooling_methods

        # Create pooling layers
        self.pooling_layers = nn.ModuleDict()

        for method in pooling_methods:
            if method == "attention":
                self.pooling_layers[method] = AttentivePooling(
                    in_channels, hidden_channels
                )
            elif method == "statistical":
                self.pooling_layers[method] = StatisticalPooling()
            else:
                # Use lambda to create proper modules for simple pooling
                self.pooling_layers[method] = LambdaPooling(method)

        # Gating network to select pooling method
        self.gate = nn.Sequential(
            nn.Linear(in_channels, len(pooling_methods)), nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Apply adaptive pooling."""

        # Compute gating weights based on input features
        if batch is not None:
            # Use mean features for gating decision
            mean_features = global_mean_pool(x, batch)
        else:
            mean_features = x.mean(dim=0, keepdim=True)

        gate_weights = self.gate(mean_features)

        # Apply each pooling method
        pooled_outputs = []
        output_dim = None

        for i, method in enumerate(self.pooling_methods):
            pooled = self.pooling_layers[method](x, batch=batch, **kwargs)

            # Ensure consistent shape
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)

            # Get output dimension from first pooling
            if output_dim is None:
                output_dim = pooled.shape[-1]

            # Ensure all pooled outputs have same dimension
            if pooled.shape[-1] != output_dim:
                # Project to same dimension if needed
                if not hasattr(self, f"proj_{method}"):
                    setattr(
                        self,
                        f"proj_{method}",
                        nn.Linear(pooled.shape[-1], output_dim).to(pooled.device),
                    )
                pooled = getattr(self, f"proj_{method}")(pooled)

            pooled_outputs.append(pooled)

        # Stack and weight by gate - ensure all have same shape
        try:
            pooled_stack = torch.stack(pooled_outputs, dim=1)  # [B, M, D]
        except RuntimeError as e:
            # If stacking fails, use the first pooling method
            print(f"Warning: Adaptive pooling failed to stack outputs: {e}")
            return pooled_outputs[0]

        gate_weights = gate_weights.unsqueeze(-1)  # [B, M, 1]

        # Weighted sum
        output = (pooled_stack * gate_weights).sum(dim=1)

        return output


class LambdaPooling(nn.Module):
    """Helper module for simple pooling operations."""

    def __init__(self, method: str):
        super().__init__()
        self.method = method

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, **kwargs) -> Tensor:
        if self.method == "mean":
            return (
                global_mean_pool(x, batch)
                if batch is not None
                else x.mean(dim=0, keepdim=True)
            )
        elif self.method == "max":
            return (
                global_max_pool(x, batch)
                if batch is not None
                else x.max(dim=0)[0].unsqueeze(0)
            )
        elif self.method == "sum":
            return (
                global_add_pool(x, batch)
                if batch is not None
                else x.sum(dim=0, keepdim=True)
            )
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")
