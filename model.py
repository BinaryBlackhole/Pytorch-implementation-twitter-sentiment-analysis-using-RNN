"""
Author: Sagar Chakraborty
Updated: Modern PyTorch 2.x implementation with LSTM and improvements
"""

import torch
import torch.nn as nn
from typing import Optional


class SentimentLSTM(nn.Module):
    """
    Modern LSTM-based sentiment classifier with dropout and optional bidirectionality.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        output_dim: Output dimension (1 for binary classification)
        n_layers: Number of LSTM layers (default: 2)
        bidirectional: Whether to use bidirectional LSTM (default: True)
        dropout: Dropout rate (default: 0.5)
        pad_idx: Padding index for embedding layer (default: 0)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = 0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Linear layer input size depends on bidirectionality
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, text_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            text: Input tensor of shape [batch_size, seq_len]
            text_lengths: Actual lengths of sequences (for packing)

        Returns:
            Output predictions of shape [batch_size, output_dim]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))

        # Pack sequence if lengths provided
        if text_lengths is not None:
            # Sort by length for pack_padded_sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                text_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            # Unpack
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            output, (hidden, cell) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states
        # hidden: [n_layers * n_directions, batch_size, hidden_dim]
        if self.lstm.bidirectional:
            # Take the last layer's forward and backward hidden states
            hidden_fwd = hidden[-2, :, :]  # Forward from last layer
            hidden_bwd = hidden[-1, :, :]  # Backward from last layer
            hidden_concat = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            hidden_concat = hidden[-1, :, :]

        # Pass through fully connected layers
        return self.fc(hidden_concat)


class SentimentRNN(nn.Module):
    """
    Legacy RNN-based sentiment classifier (kept for compatibility).
    For better performance, use SentimentLSTM instead.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            text: Input tensor of shape [batch_size, seq_len]

        Returns:
            Output predictions of shape [batch_size, output_dim]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(text)

        # RNN: output [batch_size, seq_len, hidden_dim], hidden [1, batch_size, hidden_dim]
        output, hidden = self.rnn(embedded)

        # Use the final hidden state for classification
        return self.fc(hidden.squeeze(0))


# Alias for backward compatibility
RNN = SentimentRNN
