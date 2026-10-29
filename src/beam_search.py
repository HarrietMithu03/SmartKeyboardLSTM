"""Custom Beam Search Decoder for LSTM Language Model.

Generates multi-token sentence continuations using hypothesis tracking,
cumulative log probabilities, length normalization, duplicate suppression,
and exact per-beam LSTM hidden state maintenance.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Dict, List, Optional, Set, Tuple
import logging

import torch
import torch.nn.functional as F

from config import BeamSearchConfig
from src.model import LSTMNeuralLanguageModel
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


@dataclass
class BeamHypothesis:
    """Internal hypothesis state during beam expansion."""
    token_ids: List[int]
    cumulative_log_prob: float
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
    is_finished: bool = False

    def length(self) -> int:
        return len(self.token_ids)

    def compute_score(self, alpha: float = 0.7, formula: Optional[str] = None) -> float:
        """Computes length-normalized log-probability score.
        
        Supports standard power normalization: score = cumulative_log_prob / (length ** alpha)
        and Google NMT length penalty: score = cumulative_log_prob / (((5 + length) / 6) ** alpha)
        """
        length = max(self.length(), 1)
        if alpha == 0.0:
            return self.cumulative_log_prob
        if formula == "gnmt":
            lp = ((5.0 + length) / 6.0) ** alpha
        else:
            lp = length ** alpha
        return self.cumulative_log_prob / lp


@dataclass
class BeamCandidate:
    """Final candidate continuation returned to the caller or interface."""
    text: str
    token_ids: List[int]
    words: List[str]
    cumulative_log_prob: float
    normalized_score: float
    is_finished: bool

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "words": self.words,
            "cumulative_log_prob": round(self.cumulative_log_prob, 4),
            "normalized_score": round(self.normalized_score, 4),
            "is_finished": self.is_finished,
        }


class BeamSearchDecoder:
    """Configurable, deterministic Beam Search decoder for recurrent language models."""

    def __init__(
        self,
        model: LSTMNeuralLanguageModel,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        config: Optional[BeamSearchConfig] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer or Tokenizer()
        self.config = config or BeamSearchConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Disallowed tokens for generation expansion
        self.banned_token_ids: Set[int] = {
            self.vocab.pad_id,
            self.vocab.unk_id,
            self.vocab.bos_id,
        }

    def decode_greedy(
        self,
        prefix_text: str,
        max_length: int = 12,
        min_length: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
        prefix_repetition_window: Optional[int] = None,
        clean_unbalanced_quotes: Optional[bool] = None,
    ) -> BeamCandidate:
        """Performs pure greedy decoding (equivalent to beam_width=1)."""
        results = self.decode(
            prefix_text=prefix_text,
            beam_width=1,
            max_length=max_length,
            min_length=min_length,
            length_penalty_alpha=0.0,
            temperature=1.0,
            no_repeat_ngram_size=no_repeat_ngram_size,
            prefix_repetition_window=prefix_repetition_window,
            clean_unbalanced_quotes=clean_unbalanced_quotes,
        )
        return results[0]

    def decode(
        self,
        prefix_text: str,
        beam_width: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        length_penalty_alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        prefix_repetition_window: Optional[int] = None,
        clean_unbalanced_quotes: Optional[bool] = None,
    ) -> List[BeamCandidate]:
        """Executes Beam Search continuation generation.
        
        Args:
            prefix_text: Prompt typed so far
            beam_width: Number of active candidate beams (k >= 1)
            max_length: Maximum tokens to generate
            min_length: Minimum tokens to generate before EOS is permitted
            length_penalty_alpha: Length normalization exponent (0.0 to 1.0)
            temperature: Softmax temperature (> 0.0)
            repetition_penalty: Multiplicative discount for previously generated tokens
            no_repeat_ngram_size: If > 0, bans generating previously seen n-grams
            prefix_repetition_window: If > 0, penalizes repeating recent prompt tokens
            clean_unbalanced_quotes: If True, cleans up isolated unclosed quotation marks
            
        Returns:
            List of unique BeamCandidate instances ranked by normalized score.
        """
        width = max(1, beam_width if beam_width is not None else self.config.beam_width)
        max_len = max(1, max_length if max_length is not None else self.config.max_length)
        min_len = max(1, min_length if min_length is not None else self.config.min_length)
        alpha = length_penalty_alpha if length_penalty_alpha is not None else self.config.length_penalty_alpha
        temp = max(1e-4, temperature if temperature is not None else self.config.temperature)
        rep_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else getattr(self.config, "no_repeat_ngram_size", 0)
        )
        prefix_window = (
            prefix_repetition_window
            if prefix_repetition_window is not None
            else getattr(self.config, "prefix_repetition_window", 0)
        )
        clean_quotes = (
            clean_unbalanced_quotes
            if clean_unbalanced_quotes is not None
            else getattr(self.config, "clean_unbalanced_quotes", False)
        )

        # Tokenize prefix
        clean_text = prefix_text.strip()
        tokens = self.tokenizer.tokenize(clean_text) if clean_text else []

        with torch.no_grad():
            # Step 1: Prime the LSTM with the prefix tokens
            prefix_penalty_ids: Set[int] = set()
            if tokens:
                context_tokens = tokens[-20:]  # Match model context length
                input_ids = self.vocab.encode(context_tokens)
                if prefix_window > 0:
                    prefix_penalty_ids = set(input_ids[-prefix_window:])
                x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                logits, initial_hidden = self.model(x)
                # Next token logits following prefix: shape (vocab_size,)
                last_logits = logits[0, -1, :].clone()
            else:
                # Empty input: initialize with zero hidden state and default token
                initial_hidden = self.model.init_hidden(batch_size=1, device=self.device)
                # Predict from unigram distribution or start
                start_x = torch.tensor([[self.vocab.token_to_id("the")]], dtype=torch.long, device=self.device)
                logits, initial_hidden = self.model(start_x, initial_hidden)
                last_logits = logits[0, -1, :].clone()

            # Mask out banned special tokens from first step
            for bid in self.banned_token_ids:
                if 0 <= bid < len(last_logits):
                    last_logits[bid] = -float("inf")

            # Prevent early termination before min_len
            if min_len > 0 and 0 <= self.vocab.eos_id < len(last_logits):
                last_logits[self.vocab.eos_id] = -float("inf")

            # Apply repetition penalty to prompt suffix tokens if configured
            if rep_penalty > 1.0 and prefix_penalty_ids:
                for pid in prefix_penalty_ids:
                    if 0 <= pid < len(last_logits):
                        if last_logits[pid] > 0:
                            last_logits[pid] /= rep_penalty
                        else:
                            last_logits[pid] *= rep_penalty

            # Apply temperature and log-softmax
            log_probs = F.log_softmax(last_logits / temp, dim=-1)

            # Select top-W candidates for initial beams
            top_log_probs, top_token_ids = torch.topk(log_probs, k=min(width, len(log_probs)))

            active_beams: List[BeamHypothesis] = []
            completed_beams: List[BeamHypothesis] = []

            for lp, tid in zip(top_log_probs.tolist(), top_token_ids.tolist()):
                is_eos = (tid == self.vocab.eos_id)
                # Clone initial hidden state for each beam branch
                h_clone = (initial_hidden[0].clone(), initial_hidden[1].clone())
                hyp = BeamHypothesis(
                    token_ids=[tid],
                    cumulative_log_prob=lp,
                    hidden=h_clone,
                    is_finished=is_eos,
                )
                if is_eos:
                    completed_beams.append(hyp)
                else:
                    active_beams.append(hyp)

            # Step 2: Auto-regressive expansion loop
            for step in range(1, max_len):
                if not active_beams:
                    break

                all_candidates: List[BeamHypothesis] = []

                for beam in active_beams:
                    # Single step forward through the model
                    last_token = torch.tensor([[beam.token_ids[-1]]], dtype=torch.long, device=self.device)
                    step_logits, next_hidden = self.model.step(last_token, beam.hidden)
                    logits_1d = step_logits[0].clone()

                    # Mask out banned tokens
                    for bid in self.banned_token_ids:
                        if 0 <= bid < len(logits_1d):
                            logits_1d[bid] = -float("inf")

                    # Prevent early termination before min_len
                    if len(beam.token_ids) < min_len and 0 <= self.vocab.eos_id < len(logits_1d):
                        logits_1d[self.vocab.eos_id] = -float("inf")

                    # Apply repetition penalty to reduce repetitive loops
                    if rep_penalty > 1.0:
                        penalty_ids = set(beam.token_ids)
                        if prefix_penalty_ids:
                            penalty_ids = penalty_ids | prefix_penalty_ids
                        for prev_tid in penalty_ids:
                            if logits_1d[prev_tid] > 0:
                                logits_1d[prev_tid] /= rep_penalty
                            else:
                                logits_1d[prev_tid] *= rep_penalty

                    # N-gram repetition blocking
                    if ngram_size > 0 and len(beam.token_ids) >= ngram_size - 1:
                        prev_ngram = tuple(beam.token_ids[-(ngram_size - 1):])
                        banned_by_ngram = set()
                        for i in range(len(beam.token_ids) - (ngram_size - 1)):
                            if tuple(beam.token_ids[i : i + (ngram_size - 1)]) == prev_ngram:
                                banned_by_ngram.add(beam.token_ids[i + (ngram_size - 1)])
                        if banned_by_ngram:
                            saved_logits = logits_1d.clone()
                            for btid in banned_by_ngram:
                                if 0 <= btid < len(logits_1d):
                                    logits_1d[btid] = -float("inf")
                            # Safety fallback if all tokens were banned
                            if torch.all(torch.isneginf(logits_1d)):
                                logits_1d = saved_logits

                    # Softmax probabilities
                    step_log_probs = F.log_softmax(logits_1d / temp, dim=-1)

                    # Top-W expansions for this beam
                    top_step_lp, top_step_indices = torch.topk(
                        step_log_probs, k=min(width, len(step_log_probs))
                    )

                    for step_lp, next_tid in zip(top_step_lp.tolist(), top_step_indices.tolist()):
                        new_tokens = beam.token_ids + [next_tid]
                        new_log_prob = beam.cumulative_log_prob + step_lp
                        is_eos = (next_tid == self.vocab.eos_id)

                        h_branch = (next_hidden[0].clone(), next_hidden[1].clone())
                        new_hyp = BeamHypothesis(
                            token_ids=new_tokens,
                            cumulative_log_prob=new_log_prob,
                            hidden=h_branch,
                            is_finished=is_eos,
                        )
                        all_candidates.append(new_hyp)

                # Rank all candidates by length-normalized score
                all_candidates.sort(key=lambda h: h.compute_score(alpha, formula="gnmt"), reverse=True)

                # Keep top-W candidates
                active_beams = []
                for candidate in all_candidates:
                    if candidate.is_finished:
                        completed_beams.append(candidate)
                    else:
                        if len(active_beams) < width:
                            active_beams.append(candidate)
                    if len(active_beams) >= width and len(completed_beams) >= width:
                        break

                active_beams = active_beams[:width]

            # Pool all candidate beams
            all_pool = completed_beams + active_beams
            if not all_pool:
                all_pool = active_beams

            # Sort by normalized score
            all_pool.sort(key=lambda h: h.compute_score(alpha, formula="gnmt"), reverse=True)

            # Deduplicate and format output
            seen_texts: Set[str] = set()
            candidates: List[BeamCandidate] = []

            for hyp in all_pool:
                # Filter special tokens from displayed continuation words
                words = [
                    self.vocab.id_to_token(tid)
                    for tid in hyp.token_ids
                    if tid not in self.banned_token_ids and tid != self.vocab.eos_id
                ]
                continuation_text = self.tokenizer.detokenize(words)
                if not continuation_text:
                    continue

                if clean_quotes:
                    if continuation_text.count('"') % 2 != 0:
                        cleaned = re.sub(r'\s*"\s*([.,!?;:])', r'\1', continuation_text)
                        if cleaned.count('"') % 2 != 0:
                            cleaned = cleaned.replace('"', '').strip()
                            cleaned = re.sub(r'\s+', ' ', cleaned)
                        continuation_text = cleaned
                        if not continuation_text:
                            continue

                if continuation_text not in seen_texts:
                    seen_texts.add(continuation_text)
                    candidates.append(
                        BeamCandidate(
                            text=continuation_text,
                            token_ids=hyp.token_ids,
                            words=words,
                            cumulative_log_prob=hyp.cumulative_log_prob,
                            normalized_score=hyp.compute_score(alpha, formula="gnmt"),
                            is_finished=hyp.is_finished,
                        )
                    )

                if len(candidates) >= width:
                    break

            # If all candidates produced empty text, fallback to best hypothesis representation
            if not candidates and all_pool:
                best_hyp = all_pool[0]
                raw_words = [
                    self.vocab.id_to_token(tid)
                    for tid in best_hyp.token_ids
                    if tid not in self.banned_token_ids
                ]
                text = self.tokenizer.detokenize(raw_words) if raw_words else ""
                candidates.append(
                    BeamCandidate(
                        text=text,
                        token_ids=best_hyp.token_ids,
                        words=raw_words,
                        cumulative_log_prob=best_hyp.cumulative_log_prob,
                        normalized_score=best_hyp.compute_score(alpha, formula="gnmt"),
                        is_finished=best_hyp.is_finished,
                    )
                )

        return candidates

        return candidates
