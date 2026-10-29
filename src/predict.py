"""Next-word autocomplete prediction module.

Loads trained model checkpoint and vocabulary, preprocesses user input,
computes next-token probability distribution, filters special tokens,
and produces top-K autocomplete suggestions with confidence percentages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn.functional as F

from config import BEST_MODEL_FILE, ModelConfig, VOCAB_FILE
from src.model import LSTMNeuralLanguageModel
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


@dataclass
class NextWordSuggestion:
    """Represents a single next-word suggestion."""
    word: str
    probability: float
    percentage: float
    token_id: int

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "probability": round(self.probability, 4),
            "percentage": round(self.percentage, 2),
            "token_id": self.token_id,
        }


class AutocompletePredictor:
    """Inference engine for real-time next-word autocomplete."""

    def __init__(
        self,
        model: Optional[LSTMNeuralLanguageModel] = None,
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
        device: str = "cpu",
        seq_len: int = 20,
    ):
        self.device = torch.device(device)
        self.vocab = vocab or (Vocabulary.load(VOCAB_FILE) if Path(VOCAB_FILE).exists() else None)
        self.tokenizer = tokenizer or Tokenizer()
        self.seq_len = seq_len
        self.model = model

        if self.model is None and Path(BEST_MODEL_FILE).exists():
            self.load_model(BEST_MODEL_FILE)
        elif self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def load_model(self, checkpoint_path: Union[str, Path]) -> None:
        """Loads model weights and architecture from a saved checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        logger.info(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        
        cfg = checkpoint.get("config", {})
        vocab_size = checkpoint.get("vocab_size", len(self.vocab) if self.vocab else 20000)
        embed_dim = cfg.get("embed_dim", 128)
        hidden_dim = cfg.get("hidden_dim", 256)
        num_layers = cfg.get("num_layers", 1)
        dropout = cfg.get("dropout", 0.2)
        self.seq_len = cfg.get("seq_len", self.seq_len)

        self.model = LSTMNeuralLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=checkpoint.get("pad_idx", 0),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded successfully into eval mode.")

    @property
    def is_ready(self) -> bool:
        """Returns True if both vocabulary and model are initialized."""
        return self.model is not None and self.vocab is not None

    def get_fallback_unigrams(self, top_k: int = 3) -> List[NextWordSuggestion]:
        """Provides frequent starting unigrams when input is empty."""
        common = ["the", "in", "it", "a", "this", "after", "during", "he"]
        results: List[NextWordSuggestion] = []
        for word in common[:top_k]:
            tid = self.vocab.token_to_id(word) if self.vocab else 0
            results.append(
                NextWordSuggestion(
                    word=word,
                    probability=1.0 / top_k,
                    percentage=(100.0 / top_k),
                    token_id=tid,
                )
            )
        return results

    def predict_next_words(
        self,
        text_prefix: str,
        top_k: int = 3,
        temperature: float = 1.0,
    ) -> List[NextWordSuggestion]:
        """Produces top-k next word predictions given a text prefix.
        
        Args:
            text_prefix: Raw text typed so far
            top_k: Number of suggestions to return (default 3)
            temperature: Softmax temperature (> 0.0)
            
        Returns:
            List of NextWordSuggestion dataclasses
        """
        if not self.is_ready:
            raise RuntimeError("AutocompletePredictor is not ready (missing model or vocab).")

        clean_text = text_prefix.strip()
        if not clean_text:
            return self.get_fallback_unigrams(top_k=top_k)

        # Tokenize prefix
        tokens = self.tokenizer.tokenize(clean_text)
        if not tokens:
            return self.get_fallback_unigrams(top_k=top_k)

        # Context truncation: retain at most the last seq_len tokens
        context_tokens = tokens[-self.seq_len :]
        token_ids = self.vocab.encode(context_tokens)

        # Create input tensor of shape (1, context_len)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(x)
            # Logits for the next token position: shape (vocab_size,)
            next_logits = logits[0, -1, :].clone()

            # Mask out special tokens from autocomplete recommendations
            special_ids = [
                self.vocab.pad_id,
                self.vocab.unk_id,
                self.vocab.bos_id,
                self.vocab.eos_id,
            ]
            for sid in special_ids:
                if 0 <= sid < len(next_logits):
                    next_logits[sid] = -float("inf")

            # Apply temperature scaling
            safe_temp = max(temperature, 1e-4)
            scaled_logits = next_logits / safe_temp

            # Compute Softmax probabilities
            probs = F.softmax(scaled_logits, dim=-1)

            # Top-K indices and values
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

        suggestions: List[NextWordSuggestion] = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            word = self.vocab.id_to_token(idx)
            suggestions.append(
                NextWordSuggestion(
                    word=word,
                    probability=p,
                    percentage=p * 100.0,
                    token_id=idx,
                )
            )

        return suggestions

    @staticmethod
    def join_text_and_continuation(
        current_text: str,
        continuation: str,
        capitalize_sentence_start: bool = False,
    ) -> str:
        """Joins current editor buffer with a suggestion or continuation.

        Correctly handles:
        - Punctuation attachment without stray leading spaces (e.g. ',', '.', '!', '?')
        - Contraction attachment (e.g. "'s", "'ve", "n't")
        - Partial word replacement when typing mid-word
        - Optional sentence-initial capitalization after terminal punctuation
        """
        if not continuation:
            return current_text

        text_stripped = continuation.strip()
        if not text_stripped:
            return current_text

        if not current_text:
            return text_stripped + " "

        no_pre_space_punct = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
        contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "n't"}

        # Check if continuation starts with punctuation or contraction
        starts_with_punct = text_stripped[0] in no_pre_space_punct
        starts_with_contraction = any(text_stripped.startswith(c) for c in contractions)

        if starts_with_punct or starts_with_contraction:
            base = current_text.rstrip()
            # Avoid duplicate terminal punctuation if user already typed it
            if base and base[-1] in no_pre_space_punct and text_stripped.startswith(base[-1]):
                text_stripped = text_stripped[1:].strip()
                if not text_stripped:
                    return base + " "
            return base + text_stripped + " "

        # Optional capitalization after terminal sentence punctuation
        cand = text_stripped
        if capitalize_sentence_start and current_text.rstrip() and current_text.rstrip()[-1] in {".", "!", "?"}:
            if len(cand) > 0 and cand[0].islower():
                cand = cand[0].upper() + cand[1:]
            elif len(cand) > 1 and cand[0] == '"' and cand[1].islower():
                cand = cand[0] + cand[1].upper() + cand[2:]

        # If text ends with a space, append directly
        if current_text.endswith(" "):
            return current_text + cand + " "

        # If text ends with punctuation, append with a space
        if current_text[-1] in {".", ",", "!", "?", ";", ":"}:
            return current_text + " " + cand + " "

        # If user was mid-typing a single word, replace partial prefix
        parts = current_text.rstrip().split()
        if parts and len(cand.split()) == 1:
            last_part = parts[-1].lower()
            if cand.lower().startswith(last_part):
                prefix = current_text[: current_text.rfind(parts[-1])]
                return prefix + cand + " "

        return current_text + " " + cand + " "

    @staticmethod
    def apply_suggestion_to_text(
        current_text: str,
        suggestion_word: str,
        capitalize_sentence_start: bool = False,
    ) -> str:
        """Applies a selected suggestion word or continuation to current text buffer."""
        return AutocompletePredictor.join_text_and_continuation(
            current_text=current_text,
            continuation=suggestion_word,
            capitalize_sentence_start=capitalize_sentence_start,
        )
