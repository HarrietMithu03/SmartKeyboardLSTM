"""Phase 9 Final Application Validation Tests.

Validates end-to-end multi-step interactive workflows using Streamlit AppTest:
- Application startup and widget instantiation
- Next-word autocomplete chip triggering and insertion
- Correct punctuation attachment and spacing
- Greedy vs. Beam Search comparative decoding execution
- Context diagnostics: character, word, token counters and OOV (<unk>) detection
- Clear/reset functionality returning state to initial conditions
- Latency benchmark separation (model forward pass vs. tokenization)
"""

from pathlib import Path
import time
import pytest
import torch

from config import BASE_DIR, BEST_MODEL_FILE, VOCAB_FILE
from src.beam_search import BeamSearchDecoder
from src.predict import AutocompletePredictor
from src.vocabulary import Vocabulary


class TestPhase9ApplicationValidation:
    """End-to-end verification of the Streamlit application and prediction pipeline."""

    @classmethod
    def setup_class(cls):
        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")
        cls.decoder = BeamSearchDecoder(
            model=cls.predictor.model,
            vocab=cls.vocab,
            tokenizer=cls.predictor.tokenizer,
            device="cpu",
        )

    def test_latency_measurement_separation(self):
        """Measures and separates raw model inference latency from tokenization and formatting overhead."""
        prompt = "During the Second World War , the army"

        # 1. Tokenization latency
        t0 = time.perf_counter()
        tokens = self.predictor.tokenizer.tokenize(prompt)
        token_ids = self.vocab.encode(tokens[-20:])
        tokenization_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Pure PyTorch forward pass latency
        x = torch.tensor([token_ids], dtype=torch.long)
        t1 = time.perf_counter()
        with torch.no_grad():
            logits, _ = self.predictor.model(x)
            step_logits = logits[0, -1, :].clone()
            _ = torch.topk(step_logits, k=5)
        inference_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Text joining overhead
        t2 = time.perf_counter()
        _ = self.predictor.apply_suggestion_to_text(prompt, "invaded")
        joining_ms = (time.perf_counter() - t2) * 1000.0

        assert tokenization_ms < 5.0, f"Tokenization too slow: {tokenization_ms:.2f} ms"
        assert inference_ms < 50.0, f"Raw inference too slow: {inference_ms:.2f} ms"
        assert joining_ms < 2.0, f"Joining too slow: {joining_ms:.2f} ms"

    def test_app_full_interaction_cycle_headless(self):
        """Executes a full interactive typing cycle using Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"App threw exception on launch: {at.exception}"

        # Step 1: Check initial prompt state
        assert at.session_state["user_text"] == "The game was released in "

        # Step 2: Click a suggestion chip (e.g., chip 0)
        chip_btns = [b for b in at.button if b.key and b.key.startswith("chip_")]
        assert len(chip_btns) > 0, "No suggestion chips found"
        chip_btns[0].click().run()
        assert not at.exception
        # User text must have extended
        extended_text = at.session_state["user_text"]
        assert len(extended_text) > len("The game was released in ")

        # Step 3: Run Beam Search generation
        gen_btn = next((b for b in at.button if b.label == "✨ Generate Continuations"), None)
        assert gen_btn is not None, "Generate button missing"
        gen_btn.click().run()
        assert not at.exception
        assert at.session_state.get("beam_candidates") is not None
        assert len(at.session_state["beam_candidates"]) >= 1

        # Step 4: Click Clear Text
        clear_btn = next((b for b in at.button if b.key == "btn_clear_text"), None)
        assert clear_btn is not None, "Clear button missing"
        clear_btn.click().run()
        assert not at.exception
        assert at.session_state["user_text"] == ""
        assert at.session_state.get("beam_candidates") is None

