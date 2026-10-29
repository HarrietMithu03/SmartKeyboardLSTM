"""Unit tests for LSTM model architecture, forward shapes, single-step inference, and checkpointing."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from config import ModelConfig
from src.data_loader import LMSequenceDataset
from src.model import LSTMNeuralLanguageModel
from src.vocabulary import Vocabulary


class TestModelArchitecture:
    """Tests for model forward pass, shapes, and inference steps."""

    def setup_method(self):
        self.vocab_size = 100
        self.embed_dim = 32
        self.hidden_dim = 64
        self.num_layers = 1
        self.model = LSTMNeuralLanguageModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
            pad_idx=0,
        )

    def test_forward_output_shapes(self):
        batch_size = 4
        seq_len = 10
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        logits, (hn, cn) = self.model(x)

        assert logits.shape == (batch_size, seq_len, self.vocab_size)
        assert hn.shape == (self.num_layers, batch_size, self.hidden_dim)
        assert cn.shape == (self.num_layers, batch_size, self.hidden_dim)

    def test_finite_outputs(self):
        x = torch.randint(0, self.vocab_size, (2, 5))
        logits, _ = self.model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"

    def test_step_inference(self):
        batch_size = 2
        token_id = torch.tensor([[5], [10]])
        step_logits, hidden = self.model.step(token_id)

        assert step_logits.shape == (batch_size, self.vocab_size)
        assert hidden[0].shape == (self.num_layers, batch_size, self.hidden_dim)
        assert torch.isfinite(step_logits).all()

    def test_init_hidden(self):
        h0, c0 = self.model.init_hidden(batch_size=3, device=torch.device("cpu"))
        assert h0.shape == (1, 3, self.hidden_dim)
        assert (h0 == 0).all()
        assert (c0 == 0).all()

    def test_checkpoint_consistency(self, tmp_path):
        ckpt_file = tmp_path / "model_ckpt.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_size": self.vocab_size,
                "config": {"embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim, "num_layers": 1, "dropout": 0.1},
                "pad_idx": 0,
            },
            ckpt_file,
        )

        loaded_ckpt = torch.load(ckpt_file, weights_only=True)
        reloaded_model = LSTMNeuralLanguageModel(
            vocab_size=loaded_ckpt["vocab_size"],
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
        )
        reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
        self.model.eval()
        reloaded_model.eval()

        test_x = torch.randint(0, self.vocab_size, (1, 5))
        with torch.no_grad():
            out1, _ = self.model(test_x)
            out2, _ = reloaded_model(test_x)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestSequenceDataset:
    """Tests for dataset sliding window and target alignment."""

    def test_dataset_alignment(self):
        # 10 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tokens = torch.arange(10, dtype=torch.long)
        ds = LMSequenceDataset(tokens, seq_len=4, stride=4)

        # First sample: x = [0, 1, 2, 3], y = [1, 2, 3, 4]
        x0, y0 = ds[0]
        assert x0.tolist() == [0, 1, 2, 3]
        assert y0.tolist() == [1, 2, 3, 4]
        # Target must be shifted by exactly +1
        assert (y0 - x0).tolist() == [1, 1, 1, 1]
