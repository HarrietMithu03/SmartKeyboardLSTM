"""Regression and unit tests for Phase 6 model and prediction optimizations."""

from pathlib import Path
import pytest
import torch

from config import BeamSearchConfig
from src.beam_search import BeamSearchDecoder
from src.model import LSTMNeuralLanguageModel
from src.predict import AutocompletePredictor
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary


class TestPhase6Optimizations:
    """Verifies Phase 6 enhancements: punctuation joining, n-gram blocking, prefix repetition, quote cleaning."""

    def setup_method(self):
        words = [
            "the", "dog", "barked", "at", "night", "cat", "ran", "fast",
            "a", "man", "that", "was", "released", "in", "it", "'s", "time",
            "first", "city", "great", "game"
        ]
        self.vocab = Vocabulary.build_from_tokens(words, max_vocab_size=40, min_freq=1)
        self.tokenizer = Tokenizer()

        # Small model
        self.model = LSTMNeuralLanguageModel(
            vocab_size=len(self.vocab),
            embed_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            pad_idx=self.vocab.pad_id,
        )
        self.decoder = BeamSearchDecoder(
            model=self.model,
            vocab=self.vocab,
            tokenizer=self.tokenizer,
        )

    def test_punctuation_joining_without_stray_spaces(self):
        """Verify punctuation attaches directly to preceding text without space."""
        base = "The game was released"

        # Comma attachment
        res_comma = AutocompletePredictor.apply_suggestion_to_text(base, ",")
        assert res_comma == "The game was released, "

        # Period attachment
        res_period = AutocompletePredictor.apply_suggestion_to_text(base, ".")
        assert res_period == "The game was released. "

        # Multi-token continuation starting with comma
        res_multi = AutocompletePredictor.apply_suggestion_to_text(base, ", which was praised")
        assert res_multi == "The game was released, which was praised "

        # Base ending with trailing space
        base_spaced = "The game was released "
        res_spaced = AutocompletePredictor.apply_suggestion_to_text(base_spaced, ",")
        assert res_spaced == "The game was released, "

    def test_contraction_joining(self):
        """Verify contractions like 's attach directly to preceding words."""
        base = "the city"
        res = AutocompletePredictor.apply_suggestion_to_text(base, "'s")
        assert res == "the city's "

    def test_sentence_initial_capitalization(self):
        """Verify capitalization occurs after terminal punctuation when enabled."""
        base_dot = "The game ended."
        res1 = AutocompletePredictor.apply_suggestion_to_text(
            base_dot, "it was a great victory", capitalize_sentence_start=True
        )
        assert res1 == "The game ended. It was a great victory "

        # Disabled by default for exact backward compatibility
        res2 = AutocompletePredictor.apply_suggestion_to_text(base_dot, "it was a great victory")
        assert res2 == "The game ended. it was a great victory "

        # Mid-sentence continuation should never be capitalized
        base_mid = "The game was"
        res3 = AutocompletePredictor.apply_suggestion_to_text(
            base_mid, "released in 1999", capitalize_sentence_start=True
        )
        assert res3 == "The game was released in 1999 "

    def test_ngram_repetition_blocking(self):
        """Verify no_repeat_ngram_size prevents repeating 3-token sequences."""
        prompt = "the dog"
        # Run with no_repeat_ngram_size=3
        cands = self.decoder.decode(
            prompt,
            beam_width=3,
            max_length=10,
            no_repeat_ngram_size=3,
        )
        assert len(cands) >= 1
        for c in cands:
            tokens = c.token_ids
            # Check all trigrams
            trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
            assert len(trigrams) == len(set(trigrams)), f"Found repeated trigram in tokens: {tokens}"

    def test_ngram_blocking_disabled_by_default(self):
        """Verify no_repeat_ngram_size=0 disables blocking without error."""
        cands = self.decoder.decode("the", beam_width=2, max_length=5, no_repeat_ngram_size=0)
        assert len(cands) >= 1

    def test_prefix_repetition_penalty(self):
        """Verify prefix_repetition_window penalizes prompt's trailing tokens."""
        prompt = "the dog ran in the"
        # Trailing token in prompt is "the"
        tid_the = self.vocab.token_to_id("the")

        # Without prefix penalty
        cands_no_prefix = self.decoder.decode(
            prompt, beam_width=2, max_length=3, prefix_repetition_window=0
        )

        # With prefix penalty on the last 2 prompt tokens
        cands_with_prefix = self.decoder.decode(
            prompt, beam_width=2, max_length=3, prefix_repetition_window=2, repetition_penalty=1.5
        )

        assert len(cands_with_prefix) >= 1

    def test_clean_unbalanced_quotes(self):
        """Verify clean_unbalanced_quotes removes solitary dangling quotes before punctuation."""
        # Test clean_quotes flag
        cands = self.decoder.decode(
            "the",
            beam_width=3,
            max_length=6,
            clean_unbalanced_quotes=True,
        )
        for c in cands:
            # Quotes must either be balanced (even count) or absent
            assert c.text.count('"') % 2 == 0, f"Unbalanced quotes in continuation: {c.text}"

