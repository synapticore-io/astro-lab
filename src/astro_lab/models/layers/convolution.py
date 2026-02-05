"""
Graph Convolution Layers for AstroLab
=====================================

Optimized convolution layers with modern PyG features.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    MessagePassing,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import add_self_loops

from .base import BaseGraphLayer


class FlexibleGraphConv(BaseGraphLayer):
    """
    Flexible graph convolution supporting multiple conv types.

    Features:
    - Multiple convolution types (GCN, GAT, SAGE, GIN, Transformer)
    - EdgeIndex optimization
    - Configurable aggregation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "gcn",
        heads: int = 1,
        concat: bool = True,
        edge_dim: Optional[int] = None,
        aggr: Union[str, Aggregation] = "mean",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.conv_type = conv_type.lower()
        self.heads = heads
        self.concat = concat
        self.edge_dim = edge_dim
        self.dropout = dropout

        # Create the appropriate convolution
        if self.conv_type == "gcn":
            self.conv = GCNConv(
                in_channels,
                out_channels,
                improved=kwargs.get("improved", False),
                cached=kwargs.get("cached", False),
                add_self_loops=kwargs.get("add_self_loops", True),
                normalize=kwargs.get("normalize", True),
                bias=kwargs.get("bias", True),
            )

        elif self.conv_type == "gat":
            self.conv = GATConv(
                in_channels,
                out_channels // heads if concat else out_channels,
                heads=heads,
                concat=concat,
                negative_slope=kwargs.get("negative_slope", 0.2),
                dropout=dropout,
                add_self_loops=kwargs.get("add_self_loops", True),
                edge_dim=edge_dim,
                bias=kwargs.get("bias", True),
            )

        elif self.conv_type == "sage":
            self.conv = SAGEConv(
                in_channels,
                out_channels,
                aggr=kwargs.get("sage_aggr", aggr),
                normalize=kwargs.get("normalize", False),
                root_weight=kwargs.get("root_weight", True),
                project=kwargs.get("project", False),
                bias=kwargs.get("bias", True),
            )

        elif self.conv_type == "gin":
            nn_gin = nn.Sequential(
                nn.Linear(in_channels, out_channels * 2),
                nn.BatchNorm1d(out_channels * 2),
                nn.ReLU(),
                nn.Linear(out_channels * 2, out_channels),
            )
            self.conv = GINConv(
                nn_gin,
                eps=kwargs.get("eps", 0.0),
                train_eps=kwargs.get("train_eps", False),
            )

        elif self.conv_type == "transformer":
            self.conv = TransformerConv(
                in_channels,
                out_channels // heads if concat else out_channels,
                heads=heads,
                concat=concat,
                beta=kwargs.get("beta", False),
                dropout=dropout,
                edge_dim=edge_dim,
                bias=kwargs.get("bias", True),
                root_weight=kwargs.get("root_weight", True),
            )

        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through the convolution."""

        # Cache EdgeIndex if possible
        edge_index = self.cache_edge_index(edge_index)

        # Apply convolution
        if self.conv_type in ["gat", "transformer"] and edge_attr is not None:
            out = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs)
        else:
            out = self.conv(x, edge_index, **kwargs)

        # Handle attention weights if requested
        if return_attention_weights and self.conv_type in ["gat", "transformer"]:
            if isinstance(out, tuple):
                return out
            else:
                return out, None

        return out

    def reset_parameters(self):
        """Reset convolution parameters."""
        self.conv.reset_parameters()


class AstronomicalGraphConv(MessagePassing):
    """
    Custom graph convolution for astronomical data.

    Features:
    - Distance-aware message passing
    - Magnitude-weighted aggregation
    - Spectral feature handling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_edge_attr: bool = True,
        use_distance_encoding: bool = True,
        aggr: str = "mean",
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_attr = use_edge_attr
        self.use_distance_encoding = use_distance_encoding

        # Message network
        message_in_dim = in_channels * 2
        if use_edge_attr:
            message_in_dim += kwargs.get("edge_attr_dim", 0)
        if use_distance_encoding:
            message_in_dim += kwargs.get("distance_encoding_dim", 16)

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(out_channels * 2, out_channels),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(out_channels * 2, out_channels),
        )

        # Distance encoder if needed
        if use_distance_encoding:
            self.distance_encoder = DistanceEncoder(
                kwargs.get("distance_encoding_dim", 16)
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        edge_attr: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with astronomical features."""

        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, num_nodes=x.size(0)
        )

        # Compute edge features if positions available
        if self.use_distance_encoding and pos is not None:
            row, col = edge_index
            edge_distances = torch.norm(pos[row] - pos[col], dim=-1)
            distance_encoding = self.distance_encoder(edge_distances)

            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, distance_encoding], dim=-1)
            else:
                edge_attr = distance_encoding

        # Start message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Construct messages."""

        # Concatenate source, target, and edge features
        inputs = [x_i, x_j]
        if edge_attr is not None:
            inputs.append(edge_attr)

        msg_input = torch.cat(inputs, dim=-1)
        return self.message_net(msg_input)

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """Update node features."""
        return self.update_net(torch.cat([x, aggr_out], dim=-1))

    def reset_parameters(self):
        """Reset all parameters."""
        for module in [self.message_net, self.update_net]:
            for layer in module:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()


class DistanceEncoder(nn.Module):
    """Encode distances for use in message passing."""

    def __init__(self, encoding_dim: int = 16, max_distance: float = 1000.0):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.max_distance = max_distance

        # Learnable frequency parameters
        self.frequencies = nn.Parameter(torch.randn(encoding_dim // 2) * 0.1)

    def forward(self, distances: Tensor) -> Tensor:
        """Encode distances using sinusoidal encoding."""

        # Normalize distances
        norm_distances = distances / self.max_distance

        # Apply sinusoidal encoding
        encoding = torch.zeros(
            distances.size(0), self.encoding_dim, device=distances.device
        )

        for i in range(self.encoding_dim // 2):
            encoding[:, 2 * i] = torch.sin(norm_distances * self.frequencies[i])
            encoding[:, 2 * i + 1] = torch.cos(norm_distances * self.frequencies[i])

        return encoding
