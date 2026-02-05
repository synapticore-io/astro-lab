"""
Temporal Decoder Layers for AstroLab Models
==========================================

Temporal decoder layers for autoencoders.
"""

import torch.nn as nn
from torch import Tensor


class AdvancedTemporalDecoder(nn.Module):
    """
    Temporal decoder for autoencoders.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        seq_length: int,
        num_layers: int = 2,
        decoder_type: str = "lstm",
        bidirectional: bool = False,  # Decoders are usually unidirectional
        attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.decoder_type = decoder_type

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # RNN layers
        if decoder_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif decoder_type == "gru":
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

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

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass through temporal decoder."""

        latent.size(0)

        # Project latent to initial hidden state
        initial_hidden = self.latent_proj(latent)  # [batch, hidden_dim]

        # Create sequence by repeating the initial hidden state
        # This is a simple approach - in practice you might want more sophisticated methods
        x = initial_hidden.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # [batch, seq_len, hidden_dim]

        # RNN decoding
        rnn_out, _ = self.rnn(x)

        # Attention mechanism
        if hasattr(self, "attention_layer"):
            attn_out, _ = self.attention_layer(rnn_out, rnn_out, rnn_out)
            rnn_out = self.attention_norm(rnn_out + attn_out)

        # Output projection
        out = self.output_proj(rnn_out)  # [batch, seq_len, out_dim]

        return out

    def reset_parameters(self):
        """Reset all parameters."""
        self.latent_proj.reset_parameters()

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        if hasattr(self, "attention_layer"):
            self.attention_layer._reset_parameters()
            self.attention_norm.reset_parameters()

        self.output_proj.reset_parameters()
