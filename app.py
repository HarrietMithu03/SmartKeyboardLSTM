"""Smart Keyboard: Professional AI Writing Assistant Application.

Refined Polish & Streamlit Layout Fix:
- Header: Unclipped, positioned cleanly below toolbar with full title, subtitle, and status pill.
- Readability: Dark #172033 labels and #64748B descriptions across all sliders, expanders, and widgets.
- Color Harmony: Indigo/violet primary accent (#4F46E5) replacing all default red slider accents.
- Layout Alignment: Balanced top alignment between primary writing workspace and settings panel.
- Clean Streamlit Chrome: Hides redundant Deploy control and status actions while maintaining full local utility.
"""

from pathlib import Path
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BEST_MODEL_FILE,
    EVAL_RESULTS_FAIR_JSON,
    VOCAB_FILE,
    AppConfig,
    BeamSearchConfig,
)
from src.beam_search import BeamCandidate, BeamSearchDecoder
from src.predict import AutocompletePredictor, NextWordSuggestion
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Smart Keyboard — AI Writing Assistant",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_css() -> None:
    """Injects refined, professional CSS styling matching the AI writing assistant palette."""
    css_path = PROJECT_ROOT / "outputs" / "design" / "style_prototype.css"
    raw_css = ""
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                raw_css = f.read()
        except Exception:
            raw_css = ""

    inline_overrides = """
    /* App Canvas & Typography */
    .stApp {
        background-color: #F5F7FB !important;
        color: #172033 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Main Container: Top padding ensures header is fully visible below top toolbar */
    .block-container {
        max-width: 1240px !important;
        padding-top: 3.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Hide redundant Streamlit deploy button and toolbar chrome */
    [data-testid="stDeployButton"], [data-testid="stToolbarActions"], #MainMenu {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
    }

    /* Compact Top Header */
    .smart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.875rem;
        margin-top: 0 !important;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .smart-header-title {
        font-size: 1.625rem;
        font-weight: 700;
        color: #172033;
        letter-spacing: -0.02em;
    }
    .smart-header-subtitle {
        font-size: 0.9375rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .status-pill-ready {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        color: #16856B;
        font-size: 0.8125rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 9999px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }

    /* Surface Card Containers - Equal Top Baseline */
    .ui-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 0 !important;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    .panel-heading {
        font-size: 1.125rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.2rem;
    }
    .panel-subheading {
        font-size: 0.875rem;
        color: #64748B;
        margin-bottom: 0.75rem;
    }

    /* Ensure All Widget Labels are Dark & Readable (#172033) */
    label,
    .stSlider label,
    .stCheckbox label,
    .stSelectSlider label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Captions & Explanations (#64748B) */
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #64748B !important;
        font-size: 0.8125rem !important;
    }

    /* Slider Overrides: Replace red with Indigo (#4F46E5) */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px #EEF2FF !important;
    }
    div[data-testid="stSlider"] div[data-testid="stSliderTickBar"] {
        background-color: #E2E8F0 !important;
    }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #172033 !important;
        font-weight: 600 !important;
    }

    /* Textarea Styling */
    .stTextArea textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 1.0625rem !important;
        line-height: 1.65 !important;
        color: #172033 !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }

    /* Meta Bar Below Editor */
    .editor-meta-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8125rem;
        color: #64748B;
        margin-top: -0.25rem;
        margin-bottom: 0.75rem;
    }

    /* Out-of-Vocabulary Warning */
    .oov-alert-box {
        display: flex;
        align-items: center;
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.8125rem;
        margin-bottom: 0.875rem;
        width: 100%;
    }

    /* Expander Headers: Dark Text on Card Surface */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stExpander"] details summary p {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }

    /* Button Polish */
    .stButton button {
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton button[kind="primary"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 2px 0 rgba(79, 70, 229, 0.2) !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #4338CA !important;
        border-color: #4338CA !important;
    }
    .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #172033 !important;
    }
    .stButton button[kind="secondary"]:hover {
        border-color: #4F46E5 !important;
        color: #4F46E5 !important;
        background-color: #EEF2FF !important;
    }

    /* Continuation Card */
    .continuation-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .continuation-card:hover {
        border-color: #CBD5E1;
    }
    .continuation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .continuation-tag {
        font-size: 0.75rem;
        font-weight: 700;
        color: #4F46E5;
        background-color: #EEF2FF;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .continuation-meta {
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.75rem;
        color: #64748B;
    }
    .continuation-quote {
        font-size: 0.98rem;
        font-weight: 500;
        color: #172033;
        line-height: 1.55;
        margin-bottom: 8px;
    }

    /* Comparative Cards */
    .compare-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
    }
    .compare-card.greedy {
        border-left: 4px solid #64748B;
    }
    .compare-card.beam {
        border-left: 4px solid #4F46E5;
    }

    /* Starter Prompts */
    .starter-container {
        padding: 10px 14px;
        background-color: #EEF2FF;
        border: 1px solid #E0E7FF;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .starter-title {
        font-size: 0.8125rem;
        font-weight: 600;
        color: #4338CA;
        margin-bottom: 4px;
    }
    """

    st.markdown(f"<style>{raw_css}\n{inline_overrides}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing language model weights and vocabulary...")
def get_predictor_and_decoder() -> Tuple[Optional[AutocompletePredictor], Optional[BeamSearchDecoder]]:
    """Cached loader for the trained PyTorch model, vocabulary, and decoders."""
    if not BEST_MODEL_FILE.exists() or not VOCAB_FILE.exists():
        return None, None

    try:
        vocab = Vocabulary.load(VOCAB_FILE)
        predictor = AutocompletePredictor(vocab=vocab, device="cpu")
        decoder = BeamSearchDecoder(
            model=predictor.model,
            vocab=vocab,
            tokenizer=predictor.tokenizer,
            device="cpu",
        )
        return predictor, decoder
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        return None, None


def detect_oov_words(text: str, tokenizer: Tokenizer, vocab: Vocabulary) -> List[str]:
    """Scans user text to detect any tokens that resolve to the <unk> index."""
    if not text.strip():
        return []
    tokens = tokenizer.tokenize(text.strip())
    oov: List[str] = []
    seen = set()
    for tok in tokens:
        if vocab.token_to_id(tok) == vocab.unk_id:
            if tok not in seen:
                oov.append(tok)
                seen.add(tok)
    return oov


def init_session_state() -> None:
    """Initializes Streamlit session state keys cleanly without race conditions."""
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = "The game was released in "
    if "editor_input" not in st.session_state:
        st.session_state["editor_input"] = st.session_state["user_text"]
    if "beam_candidates" not in st.session_state:
        st.session_state["beam_candidates"] = None
    if "beam_latency" not in st.session_state:
        st.session_state["beam_latency"] = 0.0
    if "comparative_results" not in st.session_state:
        st.session_state["comparative_results"] = None


# Session State Callback Handlers
def on_editor_text_change():
    """Triggered on manual keystroke edits inside the text area."""
    st.session_state["user_text"] = st.session_state.get("editor_input", "")


def on_clear_text_clicked():
    """Triggered when user clicks Clear Text."""
    st.session_state["user_text"] = ""
    st.session_state["editor_input"] = ""
    st.session_state["beam_candidates"] = None
    st.session_state["comparative_results"] = None


def on_example_prompt_clicked(prompt: str):
    """Triggered when an example prompt pill is selected."""
    text = prompt.strip() + " "
    st.session_state["user_text"] = text
    st.session_state["editor_input"] = text
    st.session_state["beam_candidates"] = None
    st.session_state["comparative_results"] = None


def on_suggestion_chip_clicked(word: str, predictor: AutocompletePredictor):
    """Triggered when user clicks a next-word suggestion chip."""
    current = st.session_state.get("user_text", "")
    new_text = predictor.apply_suggestion_to_text(current, word, capitalize_sentence_start=False)
    st.session_state["user_text"] = new_text
    st.session_state["editor_input"] = new_text
    st.session_state["beam_candidates"] = None
    st.session_state["comparative_results"] = None


def on_beam_continuation_clicked(cand_text: str, predictor: AutocompletePredictor):
    """Triggered when user clicks Insert Continuation from a candidate card."""
    current = st.session_state.get("user_text", "")
    new_text = predictor.apply_suggestion_to_text(current, cand_text, capitalize_sentence_start=True)
    st.session_state["user_text"] = new_text
    st.session_state["editor_input"] = new_text
    st.session_state["beam_candidates"] = None
    st.session_state["comparative_results"] = None


def render_compact_header(is_ready: bool) -> None:
    """Renders the top application header and subtle status indicator."""
    status_html = (
        '<span class="status-pill-ready">● Model loaded (Local CPU)</span>'
        if is_ready
        else '<span style="color: #DC2626; font-size: 0.85rem; font-weight: 600;">⚠️ Model offline</span>'
    )
    st.markdown(
        f"""
        <div class="smart-header">
            <div>
                <div class="smart-header-title">Smart Keyboard</div>
                <div class="smart-header-subtitle">AI-powered next-word prediction and text continuation</div>
            </div>
            <div>
                {status_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_custom_css()
    init_session_state()

    predictor, decoder = get_predictor_and_decoder()
    is_ready = predictor is not None and predictor.is_ready
    render_compact_header(is_ready)

    if not is_ready:
        st.error(
            "Trained model checkpoint (`models/best_model.pt`) or vocabulary (`models/vocab.json`) was not found. "
            "Please ensure model artifacts are present."
        )
        return

    # Two-Column Desktop Grid Layout (Primary Workspace: 68%, Settings Panel: 32%)
    col_workspace, col_settings = st.columns([2.15, 1.0], gap="large")

    # =========================================================================
    # PRIMARY WRITING WORKSPACE (LEFT COLUMN)
    # =========================================================================
    with col_workspace:
        # Top Workspace Panel: Smart Typing
        st.markdown(
            """
            <div class="ui-panel">
                <div class="panel-heading">Smart Typing</div>
                <div class="panel-subheading">Write naturally and choose a suggestion to continue.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------------------
        # Try an Example Section (Categorized Interactive Prompts)
        # ---------------------------------------------------------------------
        st.markdown("## Try an example")
        st.markdown(
            """
            <div style="font-size: 0.875rem; color: #64748B; margin-top: -0.6rem; margin-bottom: 0.25rem;">
                Choose a sample prompt to see how Smart Keyboard continues the text.
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-style: italic; margin-bottom: 0.5rem;">
                Prediction quality depends on the training data and context. These examples demonstrate the model's behavior, not guaranteed completions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        example_categories = {
            "History": [
                "During the Second World War",
                "The war ended in",
                "According to historical records",
            ],
            "Science and Technology": [
                "The computer system was designed to",
                "The results of the experiment showed",
                "The technology was developed to",
            ],
            "Geography": [
                "The city is located in the",
                "The river flows through the",
                "The university was founded in",
            ],
            "Academic Writing": [
                "The main reason for this was",
                "The purpose of this study is",
                "The results indicate that",
            ],
            "Everyday Writing": [
                "Please let me know if you",
                "Thank you for your",
                "I would like to know more about",
            ],
            "Gaming and Entertainment": [
                "The game was released in",
                "The film was released in",
                "The album was recorded in",
            ],
        }

        tab_names = list(example_categories.keys())
        example_tabs = st.tabs(tab_names)
        for tab_idx, (cat_name, p_list) in enumerate(example_categories.items()):
            with example_tabs[tab_idx]:
                p_cols = st.columns(len(p_list))
                for p_idx, prompt_text in enumerate(p_list):
                    with p_cols[p_idx]:
                        btn_key = f"ex_btn_{tab_idx}_{p_idx}" if tab_idx > 0 else f"ex_btn_{p_idx}"
                        st.button(
                            prompt_text,
                            key=btn_key,
                            on_click=on_example_prompt_clicked,
                            args=(prompt_text,),
                            use_container_width=True,
                            help=f"Load '{prompt_text}' into editor",
                        )

        # Large Comfortable Text Editor
        st.text_area(
            label="Text Editor",
            key="editor_input",
            on_change=on_editor_text_change,
            height=160,
            placeholder="Start typing your text here...",
            label_visibility="collapsed",
            help="Type here. Next-word suggestions update automatically as you type.",
        )

        # Synchronize text state
        current_text = st.session_state.get("editor_input", "")
        st.session_state["user_text"] = current_text

        # Character, Word, and Context Counters
        char_count = len(current_text)
        word_count = len(current_text.split()) if current_text.strip() else 0
        prompt_tokens = predictor.tokenizer.tokenize(current_text) if current_text.strip() else []
        token_count = len(prompt_tokens)
        context_used = min(token_count, predictor.seq_len)

        col_meta, col_actions = st.columns([2.2, 1])
        with col_meta:
            st.markdown(
                f"""
                <div class="editor-meta-bar">
                    <span><b>{char_count}</b> characters &nbsp;•&nbsp; <b>{word_count}</b> words &nbsp;•&nbsp; Context: <b>{context_used} / {predictor.seq_len}</b> tokens</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_actions:
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                st.button(
                    "Clear",
                    key="btn_clear_text",
                    on_click=on_clear_text_clicked,
                    use_container_width=True,
                )
            with act_col2:
                if st.button("Copy", key="btn_copy_text", use_container_width=True):
                    st.toast("Text copied to clipboard buffer!", icon="📋")

        # Out-of-Vocabulary (<unk>) Detection Banner
        oov_words = detect_oov_words(current_text, predictor.tokenizer, predictor.vocab)
        if oov_words:
            oov_formatted = ", ".join(f"'{w}'" for w in oov_words[:3])
            if len(oov_words) > 3:
                oov_formatted += f" (+{len(oov_words)-3} more)"
            st.markdown(
                f"""
                <div class="oov-alert-box">
                    <span>⚠️ <b>Unfamiliar word:</b> {oov_formatted} — mapped to <code>&lt;unk&gt;</code> in model context.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # Next-Word Suggestions Ribbon (Immediately beneath editor)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div style="font-weight: 700; color: #172033; font-size: 0.95rem; margin-top: 0.25rem; margin-bottom: 0.2rem;">Next-word suggestions</div>
            <div style="font-size: 0.8125rem; color: #64748B; margin-bottom: 0.5rem;">Click any suggestion chip to insert it seamlessly into your text:</div>
            """,
            unsafe_allow_html=True,
        )

        ac_temperature = float(st.session_state.get("slider_temperature", st.session_state.get("cfg_temperature", 1.0)))
        t_ac_start = time.perf_counter()
        try:
            suggestions = predictor.predict_next_words(
                current_text,
                top_k=5,
                temperature=ac_temperature,
            )
            ac_latency_ms = (time.perf_counter() - t_ac_start) * 1000.0
        except Exception as e:
            logger.error(f"Autocomplete error: {e}", exc_info=True)
            suggestions = []
            ac_latency_ms = 0.0

        if suggestions:
            chip_cols = st.columns(5)
            for idx, sug in enumerate(suggestions):
                with chip_cols[idx]:
                    # Distinction for top candidate
                    if idx == 0:
                        btn_label = f"★ {sug.word}\n{sug.percentage:.1f}%"
                    else:
                        btn_label = f"{sug.word}\n{sug.percentage:.1f}%"

                    st.button(
                        btn_label,
                        key=f"chip_{idx}_{sug.word}_{idx}",
                        on_click=on_suggestion_chip_clicked,
                        args=(sug.word, predictor),
                        use_container_width=True,
                        help=f"Insert '{sug.word}' (Probability: {sug.probability:.4f})",
                    )

            st.caption(f"⚡ Autocomplete inference latency: **{ac_latency_ms:.1f} ms** | Temperature: **{ac_temperature:.1f}** (Local CPU)")
        else:
            st.info("No suggestions available for this context.")

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Continuation Studio ("Continue Writing")
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="panel-heading">Continue Writing</div>
            <div class="panel-subheading">Generate coherent multi-word sentence branches exploring hypothesis paths.</div>
            """,
            unsafe_allow_html=True,
        )

        gen_col1, gen_col2 = st.columns([1.5, 3])
        with gen_col1:
            btn_generate = st.button(
                "✨ Generate Continuations",
                key="btn_generate_beam",
                type="primary",
                use_container_width=True,
            )

        # Retrieve settings
        cfg_width = st.session_state.get("cfg_beam_width", 3)
        cfg_len = st.session_state.get("cfg_max_length", 8)
        cfg_temp = float(st.session_state.get("slider_temperature", st.session_state.get("cfg_temperature", 1.0)))
        cfg_alpha = st.session_state.get("cfg_length_alpha", 0.7)
        cfg_rep = st.session_state.get("cfg_rep_penalty", 1.15)
        cfg_ngram = st.session_state.get("cfg_ngram_size", 3)
        cfg_prefix = st.session_state.get("cfg_prefix_window", 3)
        cfg_quotes = st.session_state.get("cfg_clean_quotes", True)

        if btn_generate:
            with st.spinner("Generating continuations with Beam Search..."):
                try:
                    t_gen = time.perf_counter()
                    candidates = decoder.decode(
                        prefix_text=current_text,
                        beam_width=cfg_width,
                        max_length=cfg_len,
                        length_penalty_alpha=cfg_alpha,
                        temperature=cfg_temp,
                        repetition_penalty=cfg_rep,
                        no_repeat_ngram_size=cfg_ngram,
                        prefix_repetition_window=cfg_prefix,
                        clean_unbalanced_quotes=cfg_quotes,
                    )
                    gen_elapsed_ms = (time.perf_counter() - t_gen) * 1000.0

                    st.session_state["beam_candidates"] = [c.to_dict() for c in candidates]
                    st.session_state["beam_latency"] = gen_elapsed_ms
                except Exception as e:
                    logger.error(f"Beam search error: {e}", exc_info=True)
                    st.error(f"Generation failed: {e}")
                    st.session_state["beam_candidates"] = None

        # Display Continuation Cards
        cands = st.session_state.get("beam_candidates")
        beam_latency = st.session_state.get("beam_latency", 0.0)

        if cands:
            st.caption(f"Showing **{len(cands)}** candidate continuations (Generation time: **{beam_latency:.1f} ms**):")
            for i, c in enumerate(cands):
                cand_text = c.get("text", "")
                score = c.get("normalized_score", 0.0)
                log_prob = c.get("cumulative_log_prob", 0.0)

                st.markdown(
                    f"""
                    <div class="continuation-card">
                        <div class="continuation-header">
                            <span class="continuation-tag">Option {i+1}</span>
                            <span class="continuation-meta">Score: <b>{score:.3f}</b> &nbsp;|&nbsp; Log-P: <b>{log_prob:.2f}</b></span>
                        </div>
                        <div class="continuation-quote">"{cand_text}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Insert Option {i+1}",
                    key=f"insert_opt_{i}_{hash(cand_text)}",
                    on_click=on_beam_continuation_clicked,
                    args=(cand_text, predictor),
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div style="background-color: #FFFFFF; border: 1.5px dashed #E2E8F0; border-radius: 10px; padding: 20px; text-align: center; color: #64748B;">
                    Click <b>Generate Continuations</b> above to explore multi-word sentence branches.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Comparative Decoding (Greedy vs. Beam Search)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="panel-heading">Comparative Decoding</div>
            <div class="panel-subheading">Side-by-side comparison of Greedy search ($W=1$) versus Beam Search ($W=3$).</div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Run Comparative Decoding", key="btn_run_compare", use_container_width=True):
            with st.spinner("Running comparative decoding audit..."):
                try:
                    # 1. Greedy search
                    tg0 = time.perf_counter()
                    greedy_cand = decoder.decode_greedy(
                        prefix_text=current_text,
                        max_length=cfg_len,
                        no_repeat_ngram_size=cfg_ngram,
                        prefix_repetition_window=cfg_prefix,
                        clean_unbalanced_quotes=cfg_quotes,
                    )
                    greedy_time = (time.perf_counter() - tg0) * 1000.0

                    # 2. Beam Search
                    tb0 = time.perf_counter()
                    beam_cands = decoder.decode(
                        prefix_text=current_text,
                        beam_width=cfg_width,
                        max_length=cfg_len,
                        length_penalty_alpha=cfg_alpha,
                        temperature=cfg_temp,
                        repetition_penalty=cfg_rep,
                        no_repeat_ngram_size=cfg_ngram,
                        prefix_repetition_window=cfg_prefix,
                        clean_unbalanced_quotes=cfg_quotes,
                    )
                    beam_time = (time.perf_counter() - tb0) * 1000.0

                    st.session_state["comparative_results"] = {
                        "greedy": {
                            "text": greedy_cand.text,
                            "score": greedy_cand.normalized_score,
                            "latency_ms": greedy_time,
                        },
                        "beam": {
                            "text": beam_cands[0].text if beam_cands else "",
                            "score": beam_cands[0].normalized_score if beam_cands else 0.0,
                            "latency_ms": beam_time,
                        },
                    }
                except Exception as e:
                    st.error(f"Comparative execution failed: {e}")

        if st.session_state.get("comparative_results"):
            cmp_res = st.session_state["comparative_results"]
            col_cmp_g, col_cmp_b = st.columns(2)
            with col_cmp_g:
                st.markdown(
                    f"""
                    <div class="compare-card greedy">
                        <div style="font-weight: 700; color: #64748B; font-size: 0.875rem;">Greedy Decoding (W=1)</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 6px;">
                            Latency: {cmp_res['greedy']['latency_ms']:.1f} ms &nbsp;|&nbsp; Score: {cmp_res['greedy']['score']:.3f}
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 500; color: #172033; line-height: 1.5;">
                            "{cmp_res['greedy']['text']}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Insert Greedy Result",
                    key="btn_insert_cmp_greedy",
                    on_click=on_beam_continuation_clicked,
                    args=(cmp_res['greedy']['text'], predictor),
                    use_container_width=True,
                )
            with col_cmp_b:
                st.markdown(
                    f"""
                    <div class="compare-card beam">
                        <div style="font-weight: 700; color: #4F46E5; font-size: 0.875rem;">Beam Search (W={cfg_width})</div>
                        <div style="font-size: 0.75rem; color: #4F46E5; margin-bottom: 6px;">
                            Latency: {cmp_res['beam']['latency_ms']:.1f} ms &nbsp;|&nbsp; Score: {cmp_res['beam']['score']:.3f}
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 500; color: #172033; line-height: 1.5;">
                            "{cmp_res['beam']['text']}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Insert Beam Result",
                    key="btn_insert_cmp_beam",
                    on_click=on_beam_continuation_clicked,
                    args=(cmp_res['beam']['text'], predictor),
                    use_container_width=True,
                )

    # =========================================================================
    # SETTINGS & MODEL INFO PANEL (RIGHT COLUMN - Aligned at top)
    # =========================================================================
    with col_settings:
        st.markdown(
            """
            <div class="ui-panel">
                <div class="panel-heading">Prediction Settings</div>
                <div class="panel-subheading">Fine-tune decoding hyperparameters.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Core Sliders with Visible Dark Labels & Explanations
        s_width = st.slider(
            "Beam width (W)",
            min_value=1,
            max_value=8,
            value=3,
            help="Number of parallel hypothesis paths retained at each step. W=1 corresponds to Greedy decoding.",
        )
        st.caption(f"Currently tracking **{s_width}** parallel path(s)")
        st.session_state["cfg_beam_width"] = s_width

        s_len = st.slider(
            "Continuation length (tokens)",
            min_value=1,
            max_value=20,
            value=8,
            help="Maximum number of tokens to generate per continuation.",
        )
        st.caption(f"Generating up to **{s_len}** tokens")
        st.session_state["cfg_max_length"] = s_len

        s_temp = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="slider_temperature",
            help="Scales logit distribution. Lower values produce more conservative, predictable text.",
        )
        st.caption(f"Temperature scaling factor: **{s_temp:.1f}**")
        st.session_state["cfg_temperature"] = s_temp

        s_alpha = st.slider(
            "Length penalty (α)",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Exponent for length normalization. Prevents penalizing longer coherent sentences.",
        )
        st.caption(f"Normalization exponent: **{s_alpha:.1f}**")
        st.session_state["cfg_length_alpha"] = s_alpha

        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

        # Advanced Decoding Settings Expander
        with st.expander("Advanced decoding settings", expanded=False):
            s_rep = st.slider(
                "Repetition penalty",
                min_value=1.0,
                max_value=2.0,
                value=1.15,
                step=0.05,
                help="Multiplicative discount applied to previously generated tokens.",
            )
            st.session_state["cfg_rep_penalty"] = s_rep

            s_ngram = st.checkbox(
                "Block 3-gram repetitions",
                value=True,
                help="Strictly prevents repeating any 3-token phrase (eliminates infinite phrase looping).",
            )
            st.session_state["cfg_ngram_size"] = 3 if s_ngram else 0

            s_prefix = st.select_slider(
                "Prefix repetition window",
                options=[0, 3, 5],
                value=3,
                help="Penalizes repeating immediate tokens from the prompt boundary at step 0.",
            )
            st.session_state["cfg_prefix_window"] = s_prefix

            s_quotes = st.checkbox(
                "Clean unbalanced quotes",
                value=True,
                help="Automatically strips unclosed solitary quotation marks before terminal punctuation.",
            )
            st.session_state["cfg_clean_quotes"] = s_quotes

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        # Subtle Model Information Expander
        with st.expander("Model Information", expanded=False):
            st.markdown(
                """
                - **Model:** 2-Layer Stacked LSTM
                - **Parameters:** ~8.1 Million (Tied Weights)
                - **Dataset:** WikiText-2 (`wikitext-2-raw-v1`)
                - **Vocabulary:** 20,000 words
                - **Inference Runtime:** Local CPU (PyTorch)
                - **Context Horizon:** 20 tokens
                - **Recorded Test Perplexity:** 222.18 *(official held-out test split)*
                """
            )


if __name__ == "__main__":
    main()
