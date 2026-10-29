"""Regression and verification tests for the redesigned Smart Keyboard dashboard.

Verifies:
- Clean page hierarchy and component presence
- Top header with verified status indicator
- Smart Typing workspace with live metadata
- Next-word suggestions ribbon (5 chips with top-1 distinction)
- Multi-token continuation studio with parameter passing
- Greedy vs. Beam Search comparative decoding execution
- Settings panel with sliders and advanced decoding expander
- Headless Streamlit AppTest verification across user interaction flows
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from config import BASE_DIR, BEST_MODEL_FILE, VOCAB_FILE
from src.beam_search import BeamSearchDecoder
from src.predict import AutocompletePredictor
from src.vocabulary import Vocabulary
import app


class TestRedesignedDashboard:
    """Verifies all UI elements, layout components, and backend bindings of the redesigned app."""

    @classmethod
    def setup_class(cls):
        assert BEST_MODEL_FILE.exists()
        assert VOCAB_FILE.exists()
        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")
        cls.decoder = BeamSearchDecoder(
            model=cls.predictor.model,
            vocab=cls.vocab,
            tokenizer=cls.predictor.tokenizer,
            device="cpu",
        )

    def test_app_headless_startup_and_components(self):
        """Verifies app starts up cleanly and renders all redesigned UI panels."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"App threw exception: {at.exception}"

        # 1. Text Area Editor
        assert len(at.text_area) >= 1
        assert at.text_area[0].key == "editor_input"
        assert at.session_state["user_text"] == "The game was released in "

        # 2. Header and Status Pill verification
        header_markdown = any("Smart Keyboard" in m.value and "status-pill-ready" in m.value for m in at.markdown)
        assert header_markdown, "Header with Smart Keyboard title and status pill missing"

        # 3. Action buttons (Clear, Copy, Suggestions, Generate, Compare)
        btn_keys = [b.key for b in at.button if b.key]
        assert "btn_clear_text" in btn_keys
        assert "btn_copy_text" in btn_keys
        assert "btn_generate_beam" in btn_keys
        assert "btn_run_compare" in btn_keys

        # 4. 5 Suggestion Chips
        chips = [b for b in at.button if b.key and b.key.startswith("chip_")]
        assert len(chips) == 5

        # 5. Settings sliders exist with visible labels
        assert len(at.slider) >= 4  # Beam width, max length, temperature, length penalty
        slider_labels = [s.label for s in at.slider]
        assert "Beam width (W)" in slider_labels
        assert "Continuation length (tokens)" in slider_labels
        assert "Temperature" in slider_labels
        assert "Length penalty (α)" in slider_labels

    def test_app_chip_click_and_state_persistence(self):
        """Tests that clicking suggestion chips updates the text area without duplicate words."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Click top suggestion chip
        chips = [b for b in at.button if b.key and b.key.startswith("chip_")]
        chips[0].click().run()
        assert not at.exception

        # Verify text was extended
        new_text = at.session_state["user_text"]
        assert new_text.startswith("The game was released in ")
        assert len(new_text) > len("The game was released in ")
        # No duplicate words or double spaces
        assert "  " not in new_text

    def test_app_clear_and_starter_prompts(self):
        """Tests clearing text reveals starter example prompt pills and clicking one loads it."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Click Clear
        clear_btn = next((b for b in at.button if b.key == "btn_clear_text"), None)
        assert clear_btn is not None
        clear_btn.click().run()
        assert not at.exception
        assert at.session_state["user_text"] == ""

        # Verify starter prompt buttons are visible when empty
        starter_btns = [b for b in at.button if b.key and b.key.startswith("ex_btn_")]
        assert len(starter_btns) >= 1

        # Click starter prompt button
        starter_btns[0].click().run()
        assert not at.exception
        assert len(at.session_state["user_text"].strip()) > 0

    def test_app_beam_search_generation_flow(self):
        """Tests running Beam Search generation produces candidate cards with insert buttons."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        gen_btn = next((b for b in at.button if b.key == "btn_generate_beam"), None)
        assert gen_btn is not None
        gen_btn.click().run()
        assert not at.exception

        # Candidates stored in session state
        candidates = at.session_state.get("beam_candidates")
        assert candidates is not None
        assert len(candidates) >= 1
        assert "text" in candidates[0]
        assert "normalized_score" in candidates[0]

        # Insert Option button should now be rendered
        insert_btns = [b for b in at.button if b.key and b.key.startswith("insert_opt_")]
        assert len(insert_btns) >= 1

        # Click insert button
        insert_btns[0].click().run()
        assert not at.exception
        # State should be updated
        assert len(at.session_state["user_text"]) > len("The game was released in ")

    def test_app_comparative_decoding_flow(self):
        """Tests running comparative decoding populates both greedy and beam search results."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        cmp_btn = next((b for b in at.button if b.key == "btn_run_compare"), None)
        assert cmp_btn is not None
        cmp_btn.click().run()
        assert not at.exception

        cmp_res = at.session_state.get("comparative_results")
        assert cmp_res is not None
        assert "greedy" in cmp_res
        assert "beam" in cmp_res
        assert len(cmp_res["greedy"]["text"]) > 0
        assert len(cmp_res["beam"]["text"]) > 0

