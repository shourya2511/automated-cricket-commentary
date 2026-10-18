"""
model.py

LSTM autoencoder model for learning pose sequence representations.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for pose sequences.
    Encodes sequences via LSTM encoder, decodes via LSTM decoder.
    """

    def __init__(self, input_size: int, hidden_size: int, seq_len: int, num_layers: int = 1):
        """
        Args:
            input_size: Number of features per timestep (34 for pose data: 17 joints * 2)
            hidden_size: Size of LSTM hidden state
            seq_len: Length of sequences
            num_layers: Number of LSTM layers (default 1)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection layer
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            reconstruction: (batch_size, seq_len, input_size)
            hidden: encoded latent representation
        """
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decode: initialize decoder with encoder's hidden/cell state
        # Start with a zero input sequence (or use the encoded representation)
        decoder_input = torch.zeros(x.size(0), self.seq_len, self.input_size, device=x.device)

        output, _ = self.decoder(decoder_input, (hidden, cell))

        # Project to input dimension
        reconstruction = self.output_proj(output)

        return reconstruction, hidden


def reconstruction_error(original, reconstructed):
    """
    Compute per-sample MSE reconstruction error.

    Args:
        original: (batch_size, seq_len, input_size)
        reconstructed: (batch_size, seq_len, input_size)

    Returns:
        errors: (batch_size,) tensor of per-sample MSE values
    """
    mse = torch.mean((original - reconstructed) ** 2, dim=(1, 2))
    return mse
