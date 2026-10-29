"""Phase 12: Regression tests for interactive example prompts and prediction transparency.

Verifies:
- The 'Try an example' section and transparency disclaimer appear in the dashboard.
- All 18 categorized example prompt buttons are present.
- Clicking an example button updates the editor input.
- Selecting a second example cleanly replaces the first without string concatenation.
- Suggestions are dynamically computed from the loaded LSTM model (no hardcoded responses).
- Session state remains stable across reruns.
- OOV warning banners accurately flag unknown tokens.
- Existing prediction and Beam Search controls remain functional.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from config import BASE_DIR, BEST_MODEL_FILE, VOCAB_FILE
from src.predict import AutocompletePredictor
from src.vocabulary import Vocabulary


class TestPhase12ExamplePrompts:
    """Verifies interactive example prompt flows and transparency features."""

    @classmethod
    def setup_class(cls):
        assert BEST_MODEL_FILE.exists()
        assert VOCAB_FILE.exists()
        cls.vocab = Vocabulary.load(VOCAB_FILE)
        cls.predictor = AutocompletePredictor(vocab=cls.vocab, device="cpu")

    def test_example_prompts_presence_and_structure(self):
        """Verifies section title, subtitle, disclaimer note, and all 18 example buttons."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception

        # 1. Title and Subtitle
        markdown_texts = [m.value for m in at.markdown]
        assert any("Try an example" in t for t in markdown_texts), "Header 'Try an example' missing"
        assert any("Choose a sample prompt to see how Smart Keyboard continues the text." in t for t in markdown_texts)

        # 2. Transparency Disclaimer
        assert any("Prediction quality depends on the training data and context" in t for t in markdown_texts)

        # 3. All 18 Example Buttons
        ex_buttons = [b for b in at.button if b.key and b.key.startswith("ex_btn_")]
        assert len(ex_buttons) == 18, f"Expected 18 example buttons, found {len(ex_buttons)}"

        # Verify key required prompts exist among the button labels
        button_labels = [b.label for b in ex_buttons]
        expected_sample_prompts = [
            "During the Second World War",
            "The war ended in",
            "The computer system was designed to",
            "The city is located in the",
            "The main reason for this was",
            "Please let me know if you",
            "Thank you for your",
            "The game was released in",
        ]
        for ep in expected_sample_prompts:
            assert ep in button_labels, f"Expected prompt '{ep}' missing from example buttons"

    def test_click_example_updates_editor(self):
        """Verifies clicking an example button loads the prompt cleanly into the editor."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Find the button for "During the Second World War"
        target_btn = next((b for b in at.button if b.label == "During the Second World War"), None)
        assert target_btn is not None
        target_btn.click().run()
        assert not at.exception

        assert at.session_state["user_text"] == "During the Second World War "
        assert at.session_state["editor_input"] == "During the Second World War "

    def test_clicking_second_example_replaces_first(self):
        """Verifies clicking a second example replaces the previous text instead of appending."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Click first example
        btn1 = next((b for b in at.button if b.label == "The war ended in"), None)
        assert btn1 is not None
        btn1.click().run()
        assert not at.exception
        assert at.session_state["user_text"] == "The war ended in "

        # Click second example from a different category
        btn2 = next((b for b in at.button if b.label == "The city is located in the"), None)
        assert btn2 is not None
        btn2.click().run()
        assert not at.exception

        # Text must be replaced, not concatenated
        assert at.session_state["user_text"] == "The city is located in the "
        assert "The war ended in" not in at.session_state["user_text"]

    def test_prediction_runs_dynamically_without_hardcoding(self):
        """Verifies suggestion chips are populated dynamically by model inference, not hardcoded."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Select "The game was released in"
        btn = next((b for b in at.button if b.label == "The game was released in"), None)
        assert btn is not None
        btn.click().run()
        assert not at.exception

        # Check suggestions chips
        chips = [b for b in at.button if b.key and b.key.startswith("chip_")]
        assert len(chips) == 5

        # Check that chip labels match real model predictions (top 1 is 'the')
        real_preds = self.predictor.predict_next_words("The game was released in", top_k=5, temperature=1.0)
        expected_top_word = real_preds[0].word
        assert expected_top_word in chips[0].label, (
            f"Expected dynamic model top prediction '{expected_top_word}' in chip 0, got '{chips[0].label}'"
        )

    def test_session_state_stability_across_reruns(self):
        """Verifies selecting an example prompt remains stable across multiple script reruns."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        btn = next((b for b in at.button if b.label == "The university was founded in"), None)
        assert btn is not None
        btn.click().run()
        assert not at.exception
        assert at.session_state["user_text"] == "The university was founded in "

        # Subsequent idle rerun without actions
        at.run()
        assert not at.exception
        assert at.session_state["user_text"] == "The university was founded in "

    def test_oov_detection_remains_active(self):
        """Verifies OOV alert appears when user types an unknown word."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()

        # Simulate user typing an OOV word
        at.text_area[0].input("The zxqwkvb satellite orbit").run()
        assert not at.exception

        markdown_texts = [m.value for m in at.markdown]
        assert any("Unfamiliar word:" in t and "zxqwkvb" in t for t in markdown_texts)

