"""Phase 8 Streamlit Application Integration & UI Workflow Tests.

Verifies:
- Text insertion and punctuation attachment
- Context diagnostics and Out-of-Vocabulary (<unk>) detection
- Supported prediction controls and parameters
- Empty input and graceful error handling
- Sequential suggestion clicks without duplication
- Comparative decoding execution (Greedy vs. Beam Search)
- Headless Streamlit AppTest execution and session state flow
"""

from pathlib import Path
import time
import pytest
import torch

from config import BASE_DIR as PROJECT_ROOT, BEST_MODEL_FILE, VOCAB_FILE
from src.beam_search import BeamSearchDecoder
from src.predict import AutocompletePredictor
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary
import app


class TestPhase8AppWorkflows:
    """Test suite for Phase 8 application logic and workflows."""

    @classmethod
    def setup_class(cls):
        assert VOCAB_FILE.exists(), f"Vocab missing: {VOCAB_FILE}"
        assert BEST_MODEL_FILE.exists(), f"Model missing: {BEST_MODEL_FILE}"

        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")
        cls.decoder = BeamSearchDecoder(
            model=cls.predictor.model,
            vocab=cls.vocab,
            tokenizer=cls.predictor.tokenizer,
            device="cpu",
        )

    def test_detect_oov_words(self):
        """Verifies OOV word detection identifies <unk> tokens without false positives."""
        known_prompt = "The game was released in North America"
        oov = app.detect_oov_words(known_prompt, self.predictor.tokenizer, self.vocab)
        assert oov == [], f"Expected no OOV words, got {oov}"

        unknown_prompt = "The zxqwkvb spacecraft flew over the zxqwkvb galaxy"
        oov2 = app.detect_oov_words(unknown_prompt, self.predictor.tokenizer, self.vocab)
        assert oov2 == ["zxqwkvb"], f"Expected ['zxqwkvb'], got {oov2}"

        assert app.detect_oov_words("", self.predictor.tokenizer, self.vocab) == []

    def test_text_insertion_and_punctuation_spacing(self):
        """Verifies clean text joining with no stray leading spaces before punctuation/contractions."""
        # Standard word
        res1 = self.predictor.apply_suggestion_to_text("The game was ", "released")
        assert res1 == "The game was released "

        # Punctuation attachment
        res2 = self.predictor.apply_suggestion_to_text("The game was released ", ",")
        assert res2 == "The game was released, "

        # Terminal period
        res3 = self.predictor.apply_suggestion_to_text("The game was released ", ".")
        assert res3 == "The game was released. "

        # Contraction
        res4 = self.predictor.apply_suggestion_to_text("It ", "'s")
        assert res4 == "It's "

        # Sentence start capitalization
        res5 = self.predictor.apply_suggestion_to_text("In 1999. ", "the team won", capitalize_sentence_start=True)
        assert res5 == "In 1999. The team won "

    def test_supported_prediction_controls(self):
        """Verifies all exposed backend parameters function without error and return candidates."""
        prompt = "The city is located in"

        # Test varying beam width
        for width in [1, 2, 4]:
            cands = self.decoder.decode(prompt, beam_width=width, max_length=5)
            assert len(cands) >= 1
            assert len(cands) <= width

        # Test full parameter sweep matching UI controls
        cands_full = self.decoder.decode(
            prefix_text=prompt,
            beam_width=3,
            max_length=6,
            temperature=0.8,
            length_penalty_alpha=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            prefix_repetition_window=3,
            clean_unbalanced_quotes=True,
        )
        assert len(cands_full) >= 1
        for c in cands_full:
            assert len(c.text.strip()) > 0
            assert c.normalized_score <= 0.0

    def test_comparative_decoding_execution(self):
        """Verifies comparative decoding produces both greedy and beam search candidates."""
        prompt = "According to the official report ,"
        greedy_c = self.decoder.decode_greedy(
            prefix_text=prompt,
            max_length=6,
            no_repeat_ngram_size=3,
            prefix_repetition_window=3,
            clean_unbalanced_quotes=True,
        )
        assert greedy_c is not None
        assert len(greedy_c.words) > 0

        beam_c = self.decoder.decode(
            prefix_text=prompt,
            beam_width=3,
            max_length=6,
            length_penalty_alpha=0.7,
            no_repeat_ngram_size=3,
            prefix_repetition_window=3,
            clean_unbalanced_quotes=True,
        )
        assert len(beam_c) >= 1

    def test_empty_input_graceful_handling(self):
        """Verifies predictor and decoder handle empty or whitespace-only input safely."""
        empty_sug = self.predictor.predict_next_words("", top_k=5)
        assert len(empty_sug) == 5
        for s in empty_sug:
            assert s.word not in ["<pad>", "<unk>", "<bos>", "<eos>"]

        ws_cands = self.decoder.decode("   ", beam_width=2, max_length=4)
        assert len(ws_cands) >= 1

    def test_sequential_chip_insertion_simulation(self):
        """Simulates sequential keyboard typing where each suggestion is inserted successively."""
        buffer = "The "
        words_to_append = ["film", "was", "directed", "by", "."]
        for w in words_to_append:
            buffer = self.predictor.apply_suggestion_to_text(buffer, w)

        assert buffer == "The film was directed by. "
        assert "  " not in buffer

    def test_headless_streamlit_apptest(self):
        """Executes the full Streamlit app headlessly using AppTest to verify rendering and rerun behavior."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        at.run()

        # Check there are no unhandled exceptions in the script run
        assert not at.exception, f"App raised exception: {at.exception}"

        # Verify main widgets rendered
        assert len(at.text_area) >= 1
        assert at.text_area[0].key == "editor_input"

        # Verify buttons exist (Clear Text, Copy, Suggestions, Generate, etc.)
        assert len(at.button) > 0

        # Simulate clicking Clear Text
        clear_btn = next((b for b in at.button if b.key == "btn_clear_text"), None)
        assert clear_btn is not None
        clear_btn.click().run()
        assert not at.exception
        assert at.session_state["user_text"] == ""

        # Simulate clicking an Example Prompt
        ex_btn = next((b for b in at.button if b.key and b.key.startswith("ex_btn_")), None)
        if ex_btn:
            ex_btn.click().run()
            assert not at.exception
            assert len(at.session_state["user_text"]) > 0
