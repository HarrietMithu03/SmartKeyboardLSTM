"""End-to-end verification test suite for Smart Keyboard LSTM.

Simulates complete user interaction flows:
- Model loading from disk
- Interactive typing and next-word suggestions
- Suggestion selection and text buffer insertion
- Beam search generation, ranking, and continuation insertion
- Edge case handling (empty input, OOV tokens, rapid typing)
- Validation of persistent metrics and plot artifacts
"""

import json
from pathlib import Path
import pytest
import torch

from config import (
    BEST_MODEL_FILE,
    EVAL_RESULTS_CSV,
    EVAL_RESULTS_JSON,
    MODEL_COMPARISON_PLOT,
    TRAIN_HISTORY_FILE,
    TRAINING_CURVES_PLOT,
    VOCAB_FILE,
)
from src.beam_search import BeamSearchDecoder
from src.predict import AutocompletePredictor
from src.vocabulary import Vocabulary


class TestEndToEndSystem:
    """Complete end-to-end test suite."""

    @classmethod
    def setup_class(cls):
        # 1. Verify artifacts exist on disk
        assert VOCAB_FILE.exists(), f"Vocabulary missing: {VOCAB_FILE}"
        assert BEST_MODEL_FILE.exists(), f"Trained checkpoint missing: {BEST_MODEL_FILE}"
        assert EVAL_RESULTS_JSON.exists(), f"Evaluation results missing: {EVAL_RESULTS_JSON}"
        assert EVAL_RESULTS_CSV.exists(), f"Evaluation CSV missing: {EVAL_RESULTS_CSV}"
        assert TRAINING_CURVES_PLOT.exists(), f"Training curves plot missing: {TRAINING_CURVES_PLOT}"
        assert MODEL_COMPARISON_PLOT.exists(), f"Comparison plot missing: {MODEL_COMPARISON_PLOT}"

        # 2. Instantiate predictor and decoder
        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")
        cls.decoder = BeamSearchDecoder(
            model=cls.predictor.model,
            vocab=cls.vocab,
            tokenizer=cls.predictor.tokenizer,
            device="cpu",
        )

    def test_e2e_next_word_autocomplete_flow(self):
        """Simulate user typing a prompt and selecting a suggestion."""
        prompt = "The game was released in"
        suggestions = self.predictor.predict_next_words(prompt, top_k=3)

        assert len(suggestions) == 3
        # Ensure descending probabilities
        assert suggestions[0].probability >= suggestions[1].probability >= suggestions[2].probability

        # Simulate user clicking first suggestion chip
        chosen_word = suggestions[0].word
        updated_text = AutocompletePredictor.apply_suggestion_to_text(prompt, chosen_word)

        assert updated_text.startswith(prompt)
        assert chosen_word in updated_text
        assert updated_text.endswith(" ")

    def test_e2e_beam_search_flow(self):
        """Simulate user requesting Beam Search completions and inserting a continuation."""
        prompt = "During the Second World War ,"
        candidates = self.decoder.decode(
            prefix_text=prompt,
            beam_width=3,
            max_length=8,
            length_penalty_alpha=0.7,
        )

        assert len(candidates) >= 1
        assert len(candidates) <= 3

        # Verify candidate scoring and text
        for c in candidates:
            assert len(c.text.strip()) > 0
            assert c.normalized_score <= 0.0

        # Simulate user inserting first continuation
        top_continuation = candidates[0].text
        new_buffer = prompt.rstrip() + " " + top_continuation + " "
        assert prompt in new_buffer
        assert top_continuation in new_buffer

    def test_e2e_empty_and_oov_resilience(self):
        """Verify the system remains usable and safe across edge cases."""
        # Empty input
        empty_sug = self.predictor.predict_next_words("", top_k=3)
        assert len(empty_sug) == 3
        empty_cands = self.decoder.decode("", beam_width=2, max_length=4)
        assert len(empty_cands) >= 1

        # OOV words
        oov_text = "Xylophonic quasar detected"
        oov_sug = self.predictor.predict_next_words(oov_text, top_k=3)
        assert len(oov_sug) == 3
        for s in oov_sug:
            assert s.word not in ["<pad>", "<unk>", "<bos>", "<eos>"]

    def test_e2e_evaluation_metrics_validity(self):
        """Verify that evaluation JSON contains genuine, positive, non-trivial metrics."""
        with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "accuracy_metrics" in data
        lstm_metrics = data["accuracy_metrics"]["LSTM"]
        ngram_metrics = data["accuracy_metrics"]["Ngram_Baseline"]

        assert lstm_metrics["top1_accuracy"] > 5.0
        assert lstm_metrics["top3_accuracy"] > 15.0
        assert lstm_metrics["top5_accuracy"] > 20.0
        assert lstm_metrics["avg_latency_ms"] < 100.0  # Real-time responsiveness < 100ms

        assert data["test_perplexity"] > 1.0
        assert data["test_loss"] > 0.0
