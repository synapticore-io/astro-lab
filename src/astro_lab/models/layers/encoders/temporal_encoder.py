"""
Temporal Encoder Layers for AstroLab Models
==========================================

Temporal encoding layers for time series and sequential data.
"""

import torch.nn as nn
from torch import Tensor


class AdvancedTemporalEncoder(nn.Module):
    """
    Temporal encoder with attention mechanisms.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        bidirectional: bool = True,
        attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.attention = attention

        # RNN layers
        if encoder_type == "lstm":
            self.rnn = nn.LSTM(
                in_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "gru":
            self.rnn = nn.GRU(
                in_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Calculate RNN output dimension
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention mechanism
        if attention:
            self.attention_layer = nn.MultiheadAttention(
                rnn_out_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(rnn_out_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim // 2),
            nn.LayerNorm(rnn_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim // 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through temporal encoder."""

        # RNN encoding
        rnn_out, _ = self.rnn(x)

        # Attention mechanism
        if self.attention:
            attn_out, _ = self.attention_layer(rnn_out, rnn_out, rnn_out)
            rnn_out = self.attention_norm(rnn_out + attn_out)

        # Global pooling over time dimension
        if self.bidirectional:
            # Use mean pooling for bidirectional
            temporal_features = rnn_out.mean(dim=1)
        else:
            # Use last time step for unidirectional
            temporal_features = rnn_out[:, -1]

        # Output projection
        out = self.output_proj(temporal_features)

        return out

    def reset_parameters(self):
        """Reset all parameters."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        if self.attention:
            self.attention_layer._reset_parameters()
            self.attention_norm.reset_parameters()

        self.output_proj.reset_parameters()
