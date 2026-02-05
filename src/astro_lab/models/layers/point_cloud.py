"""
Modern Point Cloud Layers for Astronomical Data
==============================================

Integration of state-of-the-art PyG point cloud operators
optimized for 50M+ astronomical objects.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.nn import (
    DynamicEdgeConv,
    GravNetConv,
    PointGNNConv,
    PointNetConv,
    PointTransformerConv,
    PPFConv,
    XConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.typing import OptTensor

from .base import BaseGraphLayer


class AstroPointCloudLayer(BaseGraphLayer):
    """
    Flexible point cloud layer supporting multiple PyG operators.

    Optimized for astronomical point clouds with:
    - Dynamic operator selection
    - Multi-scale processing
    - Memory efficiency for 50M+ objects
    """

    SUPPORTED_LAYERS = {
        "pointnet": PointNetConv,
        "dynamic_edge": DynamicEdgeConv,
        "point_transformer": PointTransformerConv,
        "gravnet": GravNetConv,
        "xconv": XConv,
        "ppf": PPFConv,
        "pointgnn": PointGNNConv,
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_type: str = "point_transformer",
        k: int = 20,
        aggr: str = "max",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()  # Call base class without parameters

        # Store dimensions as instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_type = layer_type
        self.k = k
        self.aggr = aggr
        self.dropout = dropout

        # Build the appropriate layer
        self.conv = self._build_layer(layer_type, **kwargs)

        # Post-processing
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

    def _build_layer(self, layer_type: str, **kwargs) -> nn.Module:
        """Build the specified point cloud layer."""

        if layer_type not in self.SUPPORTED_LAYERS:
            raise ValueError(f"Unknown layer type: {layer_type}")

        layer_class = self.SUPPORTED_LAYERS[layer_type]

        # Layer-specific configurations
        if layer_type == "pointnet":
            # PointNet layer with local and global features
            local_nn = nn.Sequential(
                nn.Linear(self.in_channels + 3, 64),  # +3 for coordinates
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.out_channels),
            )
            global_nn = nn.Sequential(
                nn.Linear(self.out_channels + 3, 256),
                nn.ReLU(),
                nn.Linear(256, self.out_channels),
            )
            return layer_class(local_nn, global_nn)

        elif layer_type == "dynamic_edge":
            # Dynamic Edge Conv for adaptive neighborhoods
            nn_model = nn.Sequential(
                nn.Linear(2 * (self.in_channels + 3), 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.out_channels),
            )
            return layer_class(nn_model, k=self.k, aggr=self.aggr)

        elif layer_type == "point_transformer":
            # Point Transformer with attention - simplified
            return layer_class(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                pos_nn=nn.Sequential(
                    nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, self.out_channels)
                ),
                attn_nn=nn.Sequential(
                    nn.Linear(self.out_channels, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.out_channels),
                ),
            )

        elif layer_type == "gravnet":
            # GravNet - gravity-inspired for astronomical data!
            return layer_class(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                space_dimensions=kwargs.get("space_dimensions", 4),
                propagate_dimensions=kwargs.get("propagate_dimensions", 16),
                k=self.k,
            )

        elif layer_type == "xconv":
            # XConv - transformation-invariant convolution
            return layer_class(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dim=3,  # 3D space
                kernel_size=self.k,
                hidden_channels=kwargs.get("hidden_channels", 128),
            )

        elif layer_type == "ppf":
            # Point Pair Features - rotation invariant
            local_nn = nn.Sequential(
                nn.Linear(self.in_channels + 4, 64),  # +4 for PPF features
                nn.ReLU(),
                nn.Linear(64, self.out_channels),
            )
            global_nn = nn.Sequential(
                nn.Linear(self.out_channels, 128),
                nn.ReLU(),
                nn.Linear(128, self.out_channels),
            )
            return layer_class(local_nn, global_nn)

        elif layer_type == "pointgnn":
            # PointGNN - graph neural network on points
            return layer_class(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                aggr=self.aggr,
                **kwargs,
            )

        else:
            # Fallback for unknown layer types
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex, None] = None,
        edge_attr: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass through point cloud layer.

        Args:
            x: Node features [N, in_channels]
            edge_index: Optional pre-computed edges (required by base class)
            edge_attr: Optional edge attributes (unused)
            pos: 3D positions [N, 3] (required for point cloud layers)
            batch: Batch assignment vector
            **kwargs: Additional arguments

        Returns:
            Updated node features [N, out_channels]
        """

        # For point cloud layers, we need positions
        if pos is None:
            raise ValueError("Point cloud layers require 'pos' (3D positions)")

        # Concatenate features with positions for some layers
        if self.layer_type in ["dynamic_edge", "pointnet"]:
            x_pos = torch.cat([x, pos], dim=-1)
        else:
            x_pos = x

        # Apply the convolution
        if self.layer_type == "point_transformer":
            # PointTransformer needs edge_index, create k-NN if not provided
            if edge_index is None:
                from torch_geometric.nn import knn_graph

                edge_index = knn_graph(
                    pos, k=self.k, batch=batch, flow="target_to_source"
                )
            x_out = self.conv(x, pos, edge_index)
        elif self.layer_type in ["dynamic_edge"]:
            x_out = self.conv(x_pos, batch=batch)
        elif self.layer_type == "gravnet":
            x_out = self.conv(x, batch=batch)
        elif self.layer_type == "xconv":
            x_out = self.conv(x, pos, batch=batch)
        elif self.layer_type == "ppf":
            # PPF needs both features and positions
            if edge_index is None:
                from torch_geometric.nn import knn_graph

                edge_index = knn_graph(
                    pos, k=self.k, batch=batch, flow="target_to_source"
                )
            x_out = self.conv(x, pos, edge_index, batch=batch)
        else:
            # Default: use edge_index if provided
            if edge_index is not None:
                x_out = self.conv(x_pos, edge_index)
            else:
                x_out = self.conv(x_pos, batch=batch)

        # Post-processing
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        x_out = self.dropout_layer(x_out)

        # Residual connection if dimensions match
        if x.shape[-1] == x_out.shape[-1]:
            x_out = x_out + x

        return x_out


class MultiScalePointCloudEncoder(nn.Module):
    """
    Multi-scale point cloud encoder for astronomical data.

    Processes point clouds at multiple scales using different
    operators for optimal feature extraction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 256, 512],
        output_dim: int = 1024,
        layer_types: List[str] = ["dynamic_edge", "point_transformer", "gravnet"],
        k_values: List[int] = [20, 40, 80],
        dropout: float = 0.1,
        pooling: str = "max",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.pooling = pooling

        # Build multi-scale layers
        self.layers = nn.ModuleList()

        in_channels = input_dim
        for i, (hidden_dim, layer_type, k) in enumerate(
            zip(hidden_dims, layer_types, k_values)
        ):
            layer = AstroPointCloudLayer(
                in_channels=in_channels,
                out_channels=hidden_dim,
                layer_type=layer_type,
                k=k,
                dropout=dropout,
            )
            self.layers.append(layer)
            in_channels = hidden_dim

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(sum(hidden_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with multi-scale processing.

        Returns:
            - Final features [N, output_dim] or [B, output_dim] if pooled
            - List of intermediate features at each scale
        """

        intermediate_features = []

        # Process through layers
        h = x
        for layer in self.layers:
            h = layer(h, edge_index=None, pos=pos, batch=batch)
            intermediate_features.append(h)

        # Concatenate multi-scale features
        multi_scale = torch.cat(intermediate_features, dim=-1)

        # Final projection
        output = self.final_proj(multi_scale)

        # Optional pooling for graph-level tasks
        if batch is not None and self.pooling is not None:
            if self.pooling == "max":
                output = global_max_pool(output, batch)
            elif self.pooling == "mean":
                output = global_mean_pool(output, batch)
            elif self.pooling == "add":
                output = global_add_pool(output, batch)

        return output, intermediate_features


class AdaptivePointCloudLayer(nn.Module):
    """
    Adaptive point cloud layer that selects the best operator
    based on data characteristics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        candidate_layers: List[str] = ["dynamic_edge", "point_transformer", "gravnet"],
        k: int = 20,
        use_gating: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gating = use_gating

        # Build candidate layers
        self.layers = nn.ModuleDict()
        for layer_type in candidate_layers:
            self.layers[layer_type] = AstroPointCloudLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                layer_type=layer_type,
                k=k,
            )

        # Gating network to select/weight layers
        if use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(in_channels + 3, 64),
                nn.ReLU(),
                nn.Linear(64, len(candidate_layers)),
                nn.Softmax(dim=-1),
            )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
        """
        Forward pass with adaptive layer selection.
        """

        if self.use_gating:
            # Compute gating weights based on features
            x_global = (
                global_mean_pool(x, batch)
                if batch is not None
                else x.mean(0, keepdim=True)
            )
            pos_global = (
                global_mean_pool(pos, batch)
                if batch is not None
                else pos.mean(0, keepdim=True)
            )

            gate_input = torch.cat([x_global, pos_global], dim=-1)
            gate_weights = self.gate_net(gate_input)

            # Expand weights for all nodes
            if batch is not None:
                gate_weights = gate_weights[batch]
            else:
                gate_weights = gate_weights.expand(x.size(0), -1)

            # Apply weighted sum of layer outputs
            output = torch.zeros(x.size(0), self.out_channels, device=x.device)

            for i, (layer_name, layer) in enumerate(self.layers.items()):
                layer_output = layer(x, edge_index=None, pos=pos, batch=batch)
                output += gate_weights[:, i : i + 1] * layer_output

            return output
        else:
            # Simple average of all layers
            outputs = []
            for layer in self.layers.values():
                outputs.append(layer(x, edge_index=None, pos=pos, batch=batch))
            return torch.stack(outputs, dim=0).mean(0)


class HierarchicalPointCloudProcessor(nn.Module):
    """
    Hierarchical processor for very large point clouds (50M+ points).

    Uses progressive downsampling with different operators at each level.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 256, 512],
        downsample_ratios: List[float] = [0.5, 0.25, 0.125],
        layer_types: List[str] = ["dynamic_edge", "point_transformer", "gravnet"],
        output_dim: int = 1024,
    ):
        super().__init__()

        self.downsample_ratios = downsample_ratios

        # Build hierarchical layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        in_channels = input_dim
        for hidden_dim, layer_type in zip(hidden_dims, layer_types):
            # Point cloud processing layer
            layer = AstroPointCloudLayer(
                in_channels=in_channels,
                out_channels=hidden_dim,
                layer_type=layer_type,
            )
            self.layers.append(layer)

            # Downsampling layer (learnable)
            downsample = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.downsample_layers.append(downsample)

            in_channels = hidden_dim

        # Final aggregation
        self.final_layer = nn.Sequential(
            nn.Linear(sum(hidden_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Hierarchical processing with progressive downsampling.

        Returns:
            - Final features
            - List of features and positions at each level
        """

        hierarchical_features = []
        level_outputs = []

        current_x = x
        current_pos = pos
        current_batch = batch

        for i, (layer, downsample, ratio) in enumerate(
            zip(self.layers, self.downsample_layers, self.downsample_ratios)
        ):
            # Process at current resolution
            current_x = layer(
                current_x, edge_index=None, pos=current_pos, batch=current_batch
            )

            # Store level output
            level_outputs.append(
                {
                    "features": current_x,
                    "positions": current_pos,
                    "batch": current_batch,
                }
            )

            # Downsample for next level (except last)
            if i < len(self.layers) - 1:
                # Compute importance scores
                importance_input = torch.cat([current_x, current_pos], dim=-1)
                importance_scores = downsample(importance_input).squeeze(-1)

                # Sample points based on importance
                if current_batch is not None:
                    # Handle batched data
                    new_x = []
                    new_pos = []
                    new_batch = []

                    for b in torch.unique(current_batch):
                        mask = current_batch == b
                        b_x = current_x[mask]
                        b_pos = current_pos[mask]
                        b_scores = importance_scores[mask]

                        # Sample top-k points
                        k = int(len(b_x) * ratio)
                        _, indices = torch.topk(b_scores, k)

                        new_x.append(b_x[indices])
                        new_pos.append(b_pos[indices])
                        new_batch.append(torch.full((k,), b, device=x.device))

                    current_x = torch.cat(new_x, dim=0)
                    current_pos = torch.cat(new_pos, dim=0)
                    current_batch = torch.cat(new_batch, dim=0)
                else:
                    # Unbatched data
                    k = int(len(current_x) * ratio)
                    _, indices = torch.topk(importance_scores, k)
                    current_x = current_x[indices]
                    current_pos = current_pos[indices]

            # Aggregate features
            if current_batch is not None:
                pooled = global_mean_pool(current_x, current_batch)
            else:
                pooled = current_x.mean(0, keepdim=True)
            hierarchical_features.append(pooled)

        # Combine hierarchical features
        combined = torch.cat(hierarchical_features, dim=-1)
        output = self.final_layer(combined)

        return output, level_outputs


def create_point_cloud_encoder(
    input_dim: int,
    output_dim: int,
    encoder_type: str = "multi_scale",
    num_objects: int = 1_000_000,
    **kwargs,
) -> nn.Module:
    """
    Factory function for creating point cloud encoders.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        encoder_type: Type of encoder ("multi_scale", "adaptive", "hierarchical")
        num_objects: Expected number of objects (for optimization)
        **kwargs: Additional arguments for the encoder

    Returns:
        Point cloud encoder module
    """

    if num_objects > 10_000_000:
        # Very large scale - use hierarchical
        encoder_type = "hierarchical"
        kwargs.setdefault("downsample_ratios", [0.1, 0.05, 0.01])
        kwargs.setdefault(
            "layer_types", ["dynamic_edge", "gravnet", "point_transformer"]
        )

    if encoder_type == "multi_scale":
        # Extract correct parameters for MultiScalePointCloudEncoder
        encoder_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": kwargs.get("hidden_dims", [128, 256, 512]),
            "layer_types": kwargs.get(
                "layer_types", ["dynamic_edge", "point_transformer", "gravnet"]
            ),
            "k_values": kwargs.get("k_values", [20, 40, 80]),
            "dropout": kwargs.get("dropout", 0.1),
            "pooling": kwargs.get("pooling", None),
        }
        return MultiScalePointCloudEncoder(**encoder_kwargs)

    elif encoder_type == "adaptive":
        # Extract correct parameters for AdaptivePointCloudLayer
        encoder_kwargs = {
            "in_channels": input_dim,
            "out_channels": output_dim,
            "candidate_layers": kwargs.get(
                "layer_types", ["dynamic_edge", "point_transformer", "gravnet"]
            ),
            "k": kwargs.get("k_neighbors", [20])[0]
            if isinstance(kwargs.get("k_neighbors", 20), list)
            else kwargs.get("k_neighbors", 20),
            "use_gating": kwargs.get("use_gating", True),
        }
        return AdaptivePointCloudLayer(**encoder_kwargs)

    elif encoder_type == "hierarchical":
        # Extract correct parameters for HierarchicalPointCloudProcessor
        encoder_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": kwargs.get("hidden_dims", [128, 256, 512]),
            "downsample_ratios": kwargs.get("downsample_ratios", [0.5, 0.25, 0.125]),
            "layer_types": kwargs.get(
                "layer_types", ["dynamic_edge", "point_transformer", "gravnet"]
            ),
        }
        return HierarchicalPointCloudProcessor(**encoder_kwargs)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
