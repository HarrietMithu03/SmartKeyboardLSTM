"""PyTorch LSTM Language Model architecture.

Implements a word-level Recurrent Neural Network Language Model with:
- Token embeddings with padding index handling
- Configurable multi-layer LSTM cell
- Regularization via dropout
- Projection head to vocabulary logits
- Utilities for forward evaluation, hidden state initialization, and single-step inference.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMNeuralLanguageModel(nn.Module):
    """LSTM-based word language model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.pad_idx = pad_idx
        self.tie_weights = tie_weights

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        
        self.drop_in = nn.Dropout(dropout)

        # PyTorch requires dropout=0 when num_layers=1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.drop_out = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Optional weight tying if embed_dim == hidden_dim
        if tie_weights:
            if embed_dim != hidden_dim:
                raise ValueError("Cannot tie weights when embed_dim != hidden_dim")
            self.fc_out.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights using standard normal / uniform distributions."""
        init_range = 0.08
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        # Ensure padding embedding remains all zeros
        if self.pad_idx is not None and self.pad_idx < self.vocab_size:
            with torch.no_grad():
                self.embedding.weight[self.pad_idx].fill_(0)

        nn.init.zeros_(self.fc_out.bias)
        nn.init.uniform_(self.fc_out.weight, -init_range, init_range)

        # Initialize LSTM hidden weights orthogonally, forget bias to 1.0
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias initialization to 1.0 for LSTM stability
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns initialized zero hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through embedding, LSTM, and linear projection.
        
        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len)
            hidden: Optional tuple (h_0, c_0)
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            hidden: Updated tuple (h_n, c_n)
        """
        # Embeddings: (batch_size, seq_len, embed_dim)
        embeds = self.drop_in(self.embedding(x))

        # LSTM: output shape (batch_size, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Dropout and linear projection: (batch_size, seq_len, vocab_size)
        out = self.drop_out(lstm_out)
        logits = self.fc_out(out)

        return logits, hidden

    def step(
        self,
        token_id: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step inference forward pass for auto-regressive generation.
        
        Args:
            token_id: Tensor of shape (batch_size, 1) or (batch_size,)
            hidden: Hidden states from preceding step
            
        Returns:
            logits: Shape (batch_size, vocab_size)
            hidden: Updated hidden states (h, c)
        """
        if token_id.ndim == 1:
            token_id = token_id.unsqueeze(1)
            
        logits, hidden = self.forward(token_id, hidden)
        return logits.squeeze(1), hidden
