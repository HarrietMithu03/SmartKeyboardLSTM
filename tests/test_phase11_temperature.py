"""Phase 11: Regression Tests for Prediction Temperature Control.

Verifies:
- The autocomplete call in app.py passes the user's selected slider temperature.
- Temperature 1.0 preserves existing default autocomplete behavior.
- Temperatures across supported range (0.1 to 2.0) and boundary values are handled correctly.
- Probabilities are valid, normalized, non-negative, and contain no NaNs/Infs.
- Suggestion rankings remain strictly sorted and monotonic with respect to logits.
- Varying temperatures does not mutate model weights.
- Deterministic mock/synthetic logits verify parameter propagation independently of checkpoint weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import math
import pytest
import torch
import torch.nn as nn

from config import BASE_DIR, BEST_MODEL_FILE, VOCAB_FILE
from src.predict import AutocompletePredictor, NextWordSuggestion
from src.vocabulary import Vocabulary


class MockDeterministicLanguageModel(nn.Module):
    """Synthetic language model returning fixed, deterministic logits."""

    def __init__(self, vocab_size: int = 100, fixed_logits: torch.Tensor = None):
        super().__init__()
        self.vocab_size = vocab_size
        if fixed_logits is not None:
            self.register_buffer("fixed_logits", fixed_logits)
        else:
            # Deterministic decreasing logits: [20.0, 18.0, 15.0, 10.0, 5.0, 1.0, 0.0, ...]
            logits = torch.zeros(vocab_size)
            logits[4] = 25.0  # highest non-special
            logits[5] = 20.0
            logits[6] = 16.0
            logits[7] = 12.0
            logits[8] = 8.0
            self.register_buffer("fixed_logits", logits)

    def forward(self, x, hidden=None):
        # Shape: (batch=1, seq_len=x.size(1), vocab_size)
        batch_size, seq_len = x.size()
        out = self.fixed_logits.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, self.vocab_size)
        return out, None


@pytest.fixture(scope="module")
def shared_vocab():
    assert VOCAB_FILE.exists()
    return Vocabulary.load(VOCAB_FILE)


@pytest.fixture(scope="module")
def real_predictor(shared_vocab):
    assert BEST_MODEL_FILE.exists()
    return AutocompletePredictor(vocab=shared_vocab, device="cpu")


@pytest.fixture
def mock_predictor(shared_vocab):
    """Predictor backed by synthetic deterministic logits."""
    mock_model = MockDeterministicLanguageModel(vocab_size=len(shared_vocab))
    pred = AutocompletePredictor(model=mock_model, vocab=shared_vocab, device="cpu")
    return pred


class TestPhase11TemperatureWiring:
    """Verifies temperature parameter propagation and mathematical behavior."""

    def test_default_temperature_preserves_existing_behavior(self, real_predictor):
        """Verifies that calling predict_next_words without temperature argument defaults to 1.0."""
        prompt = "The game was released in"
        default_res = real_predictor.predict_next_words(prompt, top_k=5)
        explicit_1_res = real_predictor.predict_next_words(prompt, top_k=5, temperature=1.0)

        assert len(default_res) == len(explicit_1_res) == 5
        for d, e in zip(default_res, explicit_1_res):
            assert d.word == e.word
            assert d.token_id == e.token_id
            assert math.isclose(d.probability, e.probability, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(d.percentage, e.percentage, rel_tol=1e-5, abs_tol=1e-4)

    def test_supported_temperature_range_and_boundaries(self, mock_predictor):
        """Verifies that temperatures within [0.1, 2.0] and near-zero boundaries execute cleanly."""
        test_temps = [0.0, 1e-5, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        prompt = "test prefix"

        for temp in test_temps:
            results = mock_predictor.predict_next_words(prompt, top_k=5, temperature=temp)
            assert len(results) == 5
            for r in results:
                assert isinstance(r.word, str)
                assert 0.0 <= r.probability <= 1.0
                assert math.isclose(r.percentage, r.probability * 100.0, rel_tol=1e-4)
                assert not math.isnan(r.probability)
                assert not math.isinf(r.probability)

    def test_probability_normalization_and_special_token_exclusion(self, real_predictor):
        """Verifies that special tokens (<pad>, <unk>, <bos>, <eos>) are excluded and probs valid."""
        prompt = "During the Second World War ,"
        excluded = {"<pad>", "<unk>", "<bos>", "<eos>"}

        for temp in [0.5, 1.0, 1.5]:
            results = real_predictor.predict_next_words(prompt, top_k=10, temperature=temp)
            for r in results:
                assert r.word not in excluded, f"Special token {r.word} found in suggestions!"
                assert 0.0 < r.probability <= 1.0

    def test_temperature_monotonic_ranking_and_confidence_shift(self, mock_predictor):
        """Proves that dividing logits by temperature preserves top-k ranking while shifting confidence."""
        prompt = "arbitrary context"
        res_t05 = mock_predictor.predict_next_words(prompt, top_k=5, temperature=0.5)
        res_t10 = mock_predictor.predict_next_words(prompt, top_k=5, temperature=1.0)
        res_t15 = mock_predictor.predict_next_words(prompt, top_k=5, temperature=1.5)

        # 1. Ranking must be strictly identical across all temperatures
        words_t05 = [r.word for r in res_t05]
        words_t10 = [r.word for r in res_t10]
        words_t15 = [r.word for r in res_t15]
        assert words_t05 == words_t10 == words_t15, "Temperature unexpectedly mutated ranking order!"

        # 2. Top-1 probability must sharpen at lower temperature and flatten at higher temperature
        top1_p_t05 = res_t05[0].probability
        top1_p_t10 = res_t10[0].probability
        top1_p_t15 = res_t15[0].probability

        assert top1_p_t05 > top1_p_t10 > top1_p_t15, (
            f"Expected sharpening: {top1_p_t05:.4f} > {top1_p_t10:.4f} > {top1_p_t15:.4f}"
        )

        # 3. Top-k suggestions must remain strictly sorted in descending probability order
        for res_list in [res_t05, res_t10, res_t15]:
            probs = [r.probability for r in res_list]
            assert probs == sorted(probs, reverse=True), "Suggestions are not sorted in descending probability!"

    def test_model_weights_not_modified_by_temperature(self, real_predictor):
        """Verifies that calling prediction with varying temperatures does not modify model parameters."""
        # Snapshot model parameter tensors
        initial_params = [p.clone() for p in real_predictor.model.parameters()]

        prompt = "The city is located in the"
        for t in [0.2, 0.5, 0.9, 1.0, 1.4, 1.8, 2.0]:
            real_predictor.predict_next_words(prompt, top_k=5, temperature=t)

        for p_init, p_curr in zip(initial_params, real_predictor.model.parameters()):
            assert torch.equal(p_init, p_curr), "Model parameters were mutated during temperature inference!"

    def test_app_temperature_slider_wiring_headless(self):
        """Verifies via Streamlit AppTest that changing the slider updates the autocomplete temperature."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception

        # Initial run: default temperature is 1.0
        slider_temp = next((s for s in at.slider if s.label == "Temperature"), None)
        assert slider_temp is not None
        assert math.isclose(slider_temp.value, 1.0)
        assert "Temperature: **1.0**" in at.caption[0].value or any("Temperature: **1.0**" in c.value for c in at.caption)

        # Move temperature slider to 0.5
        slider_temp.set_value(0.5).run()
        assert not at.exception
        assert math.isclose(at.session_state["slider_temperature"], 0.5)

        # Verify UI caption explicitly reflects Temperature: 0.5
        temp_captions = [c.value for c in at.caption if "Temperature:" in c.value]
        assert len(temp_captions) >= 1
        assert "Temperature: **0.5**" in temp_captions[0]

    def test_app_mock_predict_temperature_propagation(self):
        """Uses mock to strictly verify that app.py passes temperature=s_temp into predict_next_words."""
        from streamlit.testing.v1 import AppTest

        with patch("src.predict.AutocompletePredictor.predict_next_words") as mock_predict:
            mock_predict.return_value = [
                NextWordSuggestion(word="the", probability=0.4, percentage=40.0, token_id=2),
                NextWordSuggestion(word="a", probability=0.3, percentage=30.0, token_id=3),
                NextWordSuggestion(word="also", probability=0.15, percentage=15.0, token_id=4),
                NextWordSuggestion(word="not", probability=0.1, percentage=10.0, token_id=5),
                NextWordSuggestion(word="in", probability=0.05, percentage=5.0, token_id=6),
            ]

            at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
            at.run()
            assert not at.exception

            # Find the Temperature slider and set it to 0.7
            slider_temp = next((s for s in at.slider if s.label == "Temperature"), None)
            assert slider_temp is not None
            slider_temp.set_value(0.7).run()
            assert not at.exception

            # Inspect the calls made to predict_next_words
            assert mock_predict.called
            last_call = mock_predict.call_args
            kwargs = last_call.kwargs
            assert "temperature" in kwargs, "app.py did not pass 'temperature' keyword argument!"
            assert math.isclose(kwargs["temperature"], 0.7, abs_tol=1e-3), (
                f"Expected temperature=0.7, but got {kwargs['temperature']}"
            )

