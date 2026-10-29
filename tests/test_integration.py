"""Integration tests verifying end-to-end flow from saved artifacts to prediction, Beam Search, and UI text manipulation."""

from pathlib import Path
import pytest
import torch

from config import BEST_MODEL_FILE, EVAL_RESULTS_JSON, VOCAB_FILE
from src.beam_search import BeamSearchDecoder
from src.predict import AutocompletePredictor
from src.vocabulary import Vocabulary


class TestIntegration:
    """Integration test suite using the real trained checkpoint and vocabulary."""

    @classmethod
    def setup_class(cls):
        assert VOCAB_FILE.exists(), f"Vocabulary file missing at {VOCAB_FILE}"
        assert BEST_MODEL_FILE.exists(), f"Best model checkpoint missing at {BEST_MODEL_FILE}"

        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")
        cls.decoder = BeamSearchDecoder(
            model=cls.predictor.model,
            vocab=cls.vocab,
            tokenizer=cls.predictor.tokenizer,
            device="cpu",
        )

    def test_predictor_loaded_and_ready(self):
        assert self.predictor.is_ready
        assert len(self.predictor.vocab) == 20000

    def test_real_next_word_autocomplete(self):
        prompt = "The game was released in"
        suggestions = self.predictor.predict_next_words(prompt, top_k=3)

        assert len(suggestions) == 3
        for s in suggestions:
            assert isinstance(s.word, str)
            assert len(s.word) > 0
            # Special tokens must be filtered out
            assert s.word not in ["<pad>", "<unk>", "<bos>", "<eos>"]
            assert 0.0 <= s.probability <= 1.0
            assert 0.0 <= s.percentage <= 100.0

    def test_real_beam_search_generation(self):
        prompt = "During the Second World War ,"
        candidates = self.decoder.decode(
            prefix_text=prompt,
            beam_width=3,
            max_length=8,
            length_penalty_alpha=0.7,
        )

        assert len(candidates) >= 1
        assert len(candidates) <= 3
        for c in candidates:
            assert len(c.text.strip()) > 0
            assert c.cumulative_log_prob <= 0.0
            # Words shouldn't contain special tokens
            for w in c.words:
                assert w not in ["<pad>", "<unk>", "<bos>", "<eos>"]

    def test_text_insertion_logic(self):
        # 1. Trailing space insertion
        text1 = "The game was "
        inserted1 = AutocompletePredictor.apply_suggestion_to_text(text1, "released")
        assert inserted1 == "The game was released "

        # 2. Punctuation attachment
        text2 = "In 1999."
        inserted2 = AutocompletePredictor.apply_suggestion_to_text(text2, "the")
        assert inserted2 == "In 1999. the "

        # 3. Partial word replacement
        text3 = "The game was rele"
        inserted3 = AutocompletePredictor.apply_suggestion_to_text(text3, "released")
        assert inserted3 == "The game was released "

    def test_empty_input_handling(self):
        empty_sug = self.predictor.predict_next_words("", top_k=3)
        assert len(empty_sug) == 3
        for s in empty_sug:
            assert len(s.word) > 0

    def test_unknown_word_handling(self):
        oov_prompt = "Zxyqwbjk flew over the moon"
        suggestions = self.predictor.predict_next_words(oov_prompt, top_k=3)
        assert len(suggestions) == 3
        for s in suggestions:
            assert s.word not in ["<pad>", "<unk>", "<bos>", "<eos>"]

    def test_eval_artifacts_exist(self):
        assert EVAL_RESULTS_JSON.exists(), f"Evaluation JSON missing at {EVAL_RESULTS_JSON}"
