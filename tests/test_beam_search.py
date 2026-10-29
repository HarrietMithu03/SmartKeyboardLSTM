"""Unit tests for Beam Search decoder, greedy equivalence, scoring, and candidate constraints."""

import pytest
import torch

from src.beam_search import BeamCandidate, BeamHypothesis, BeamSearchDecoder
from src.model import LSTMNeuralLanguageModel
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary


class TestBeamSearch:
    """Tests for Beam Search decoding algorithm."""

    def setup_method(self):
        words = ["the", "dog", "barked", "at", "night", "cat", "ran", "fast", "a", "man"]
        self.vocab = Vocabulary.build_from_tokens(words, max_vocab_size=30, min_freq=1)
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

    def test_beam_width_one_equals_greedy(self):
        """Verify that beam width 1 behaves identically to greedy decoding."""
        prompt = "the dog"
        greedy_cand = self.decoder.decode_greedy(prompt, max_length=5)
        beam1_cands = self.decoder.decode(
            prompt,
            beam_width=1,
            max_length=5,
            length_penalty_alpha=0.0,
            temperature=1.0,
        )

        assert len(beam1_cands) == 1
        assert beam1_cands[0].token_ids == greedy_cand.token_ids
        assert beam1_cands[0].text == greedy_cand.text
        assert abs(beam1_cands[0].cumulative_log_prob - greedy_cand.cumulative_log_prob) < 1e-4

    def test_beam_width_limits(self):
        """Verify that decoder respects beam width cap."""
        prompt = "the"
        for w in [1, 2, 3, 5]:
            cands = self.decoder.decode(prompt, beam_width=w, max_length=4)
            assert len(cands) <= w
            assert len(cands) >= 1

    def test_cumulative_log_prob_negative(self):
        """Probabilities are in (0, 1], so log-probs must be <= 0."""
        prompt = "the"
        cands = self.decoder.decode(prompt, beam_width=2, max_length=4)
        for c in cands:
            assert c.cumulative_log_prob <= 0.0, f"Expected non-positive log prob, got {c.cumulative_log_prob}"

    def test_length_normalization_penalty(self):
        """Test hypothesis scoring with alpha penalty."""
        hyp = BeamHypothesis(
            token_ids=[10, 11, 12, 13],
            cumulative_log_prob=-8.0,
            hidden=None,
        )
        # alpha = 0.0 -> -8.0 / (4**0) = -8.0
        assert hyp.compute_score(alpha=0.0) == -8.0
        # alpha = 1.0 -> -8.0 / 4 = -2.0
        assert hyp.compute_score(alpha=1.0) == -2.0

    def test_empty_and_whitespace_prompt(self):
        """Decoder handles empty prompt gracefully without crashing."""
        cands1 = self.decoder.decode("", beam_width=2, max_length=3)
        assert len(cands1) >= 1

        cands2 = self.decoder.decode("   \n\t  ", beam_width=2, max_length=3)
        assert len(cands2) >= 1

    def test_duplicate_suppression(self):
        """Verify that returned candidate texts are distinct."""
        prompt = "the"
        cands = self.decoder.decode(prompt, beam_width=4, max_length=6)
        texts = [c.text for c in cands]
        assert len(texts) == len(set(texts)), "Decoder returned duplicate candidate texts"

    def test_banned_special_tokens(self):
        """Regression test: verify banned special tokens are never present in candidates."""
        prompt = "the dog"
        cands = self.decoder.decode(prompt, beam_width=3, max_length=6)
        for c in cands:
            for w in c.words:
                assert w not in ["<pad>", "<unk>", "<bos>", "<eos>"]
            assert "<pad>" not in c.text
            assert "<unk>" not in c.text
            assert "<bos>" not in c.text
            assert "<eos>" not in c.text

    def test_eos_termination(self):
        """Regression test: verify EOS marks candidate as finished."""
        # Force EOS to have highest probability
        orig_weights = self.model.fc_out.weight.data.clone()
        orig_bias = self.model.fc_out.bias.data.clone()
        try:
            self.model.fc_out.bias.data.fill_(-100.0)
            self.model.fc_out.bias.data[self.vocab.eos_id] = 100.0
            # With min_length=1, step 1 (second token) will pick EOS
            cands = self.decoder.decode("the", beam_width=2, min_length=1, max_length=5)
            assert len(cands) >= 1
            assert any(c.is_finished for c in cands)
        finally:
            self.model.fc_out.weight.data.copy_(orig_weights)
            self.model.fc_out.bias.data.copy_(orig_bias)

    def test_minimum_length_enforcement(self):
        """Regression test: verify EOS is prevented before min_length tokens."""
        orig_bias = self.model.fc_out.bias.data.clone()
        try:
            self.model.fc_out.bias.data.fill_(-100.0)
            self.model.fc_out.bias.data[self.vocab.eos_id] = 100.0
            # Set dog to high logit as well so valid tokens can be picked
            self.model.fc_out.bias.data[self.vocab.token_to_id("dog")] = 50.0

            min_len = 3
            cands = self.decoder.decode("the", beam_width=1, min_length=min_len, max_length=6)
            assert len(cands) == 1
            # Length must be at least min_length before EOS
            assert len(cands[0].token_ids) >= min_len
        finally:
            self.model.fc_out.bias.data.copy_(orig_bias)

    def test_maximum_length_enforcement(self):
        """Regression test: verify generation never exceeds max_length tokens."""
        for max_len in [2, 4, 7]:
            cands = self.decoder.decode("the dog", beam_width=2, max_length=max_len)
            for c in cands:
                assert len(c.token_ids) <= max_len

    def test_deterministic_decoding(self):
        """Regression test: verify identical calls produce identical candidates."""
        prompt = "the cat ran"
        res1 = self.decoder.decode(prompt, beam_width=3, max_length=5)
        res2 = self.decoder.decode(prompt, beam_width=3, max_length=5)

        assert len(res1) == len(res2)
        for c1, c2 in zip(res1, res2):
            assert c1.token_ids == c2.token_ids
            assert c1.text == c2.text
            assert abs(c1.cumulative_log_prob - c2.cumulative_log_prob) < 1e-5
            assert abs(c1.normalized_score - c2.normalized_score) < 1e-5

    def test_hidden_state_propagation_soundness(self):
        """Regression test: verify auto-regressive step unrolling matches full-sequence forward pass."""
        self.model.eval()
        tokens = ["the", "dog", "barked"]
        ids = self.vocab.encode(tokens)
        x_full = torch.tensor([ids], dtype=torch.long)

        # Full forward pass
        logits_full, hidden_full = self.model(x_full)

        # Step-by-step unrolling
        h_step = None
        logits_steps = []
        for tid in ids:
            t_tensor = torch.tensor([[tid]], dtype=torch.long)
            logit_step, h_step = self.model.step(t_tensor, h_step)
            logits_steps.append(logit_step)

        # Verify step logits match sequence logits at every position
        for i, step_logit in enumerate(logits_steps):
            assert torch.allclose(step_logit, logits_full[:, i, :], atol=1e-5)

        # Verify final hidden states match
        assert torch.allclose(h_step[0], hidden_full[0], atol=1e-5)
        assert torch.allclose(h_step[1], hidden_full[1], atol=1e-5)

    def test_punctuation_formatting_detokenizer(self):
        """Regression test: verify detokenizer punctuation formatting."""
        # Commas and periods attached to previous word
        toks1 = ["hello", ",", "world", "."]
        assert self.tokenizer.detokenize(toks1) == "hello, world."

        # Contractions attached without space
        toks2 = ["it", "'s", "time"]
        assert self.tokenizer.detokenize(toks2) == "it's time"

        # Hyphenated compound words
        toks3 = ["twenty", "-", "five"]
        assert self.tokenizer.detokenize(toks3) == "twenty-five"

        # Quotes formatting
        toks4 = ['"', "great", "game", '"']
        assert self.tokenizer.detokenize(toks4) == '"great game"'

    def test_audit_artifacts_schema_and_integrity(self):
        """Regression test: verify audit artifacts exist and adhere to contract."""
        import json
        from pathlib import Path
        prompts_p = Path("outputs/metrics/beam_audit_prompts.json")
        results_p = Path("outputs/metrics/beam_audit_results.json")

        assert prompts_p.exists(), f"Missing prompts artifact: {prompts_p}"
        assert results_p.exists(), f"Missing results artifact: {results_p}"

        with open(prompts_p, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        assert len(p_data["prompts"]) == 25
        assert "disclaimer" in p_data

        with open(results_p, "r", encoding="utf-8") as f:
            r_data = json.load(f)
        assert len(r_data["results"]) == 25
        assert "metadata" in r_data
