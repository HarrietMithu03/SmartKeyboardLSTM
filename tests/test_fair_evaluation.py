"""Regression tests for fair evaluation policy, special-token handling, and n-gram backoff."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.evaluate import NgramBaseline, evaluate_top_k_accuracy
from src.model import LSTMNeuralLanguageModel
from src.vocabulary import Vocabulary


class TestFairEvaluation:
    """Test suite verifying fair evaluation rules between LSTM and N-gram baseline."""

    def setup_method(self):
        words = ["the", "dog", "barked", "at", "night", "cat", "ran", "fast", "a", "man", "house", "car"]
        self.vocab = Vocabulary.build_from_tokens(words, max_vocab_size=30, min_freq=1)
        self.disallowed = {
            self.vocab.pad_id,
            self.vocab.unk_id,
            self.vocab.bos_id,
            self.vocab.eos_id,
        }

        # Initialize small test model
        self.model = LSTMNeuralLanguageModel(
            vocab_size=len(self.vocab),
            embed_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            pad_idx=self.vocab.pad_id,
        )
        self.model.eval()

        # Initialize baseline
        self.baseline = NgramBaseline(self.vocab)
        train_tokens = [
            self.vocab.token_to_id(w)
            for w in ["the", "dog", "barked", "at", "night", "the", "cat", "ran", "fast", "the", "dog", "ran"]
        ]
        self.baseline.fit(train_tokens)

    def test_ngram_predict_top_k_never_contains_special_tokens(self):
        """Verify N-gram baseline never returns pad, unk, bos, or eos in top-k."""
        # Add special tokens deliberately to counts
        for sp_id in self.disallowed:
            self.baseline.unigram_counts[sp_id] = 10000
            self.baseline.bigram_counts[self.vocab.token_to_id("the")][sp_id] = 10000
            self.baseline.trigram_counts[(self.vocab.token_to_id("the"), self.vocab.token_to_id("dog"))][sp_id] = 10000

        context = [self.vocab.token_to_id("the"), self.vocab.token_to_id("dog")]
        top_k_preds = self.baseline.predict_top_k(context, k=5)

        assert len(top_k_preds) > 0
        for tid in top_k_preds:
            assert tid not in self.disallowed, f"Disallowed token {tid} found in {top_k_preds}"

    def test_trigram_bigram_unigram_backoff_supplementation_without_duplicates(self):
        """Verify backoff supplements higher order candidates with lower order candidates without duplicates."""
        bl = NgramBaseline(self.vocab)
        t_the = self.vocab.token_to_id("the")
        t_dog = self.vocab.token_to_id("dog")
        t_barked = self.vocab.token_to_id("barked")
        t_ran = self.vocab.token_to_id("ran")
        t_fast = self.vocab.token_to_id("fast")
        t_cat = self.vocab.token_to_id("cat")
        t_night = self.vocab.token_to_id("night")

        # Trigram has only 1 candidate: ("the", "dog") -> "barked"
        bl.trigram_counts[(t_the, t_dog)][t_barked] = 10

        # Bigram has 2 candidates: "dog" -> "barked" (duplicate), "ran" (new)
        bl.bigram_counts[t_dog][t_barked] = 20
        bl.bigram_counts[t_dog][t_ran] = 15

        # Unigram has several candidates
        bl.unigram_counts[t_fast] = 100
        bl.unigram_counts[t_cat] = 90
        bl.unigram_counts[t_night] = 80
        bl.unigram_counts[t_ran] = 70  # Duplicate from bigram

        context = [t_the, t_dog]
        preds = bl.predict_top_k(context, k=5)

        assert len(preds) == 5
        # Verify no duplicates
        assert len(preds) == len(set(preds))
        # First should be trigram candidate: barked
        assert preds[0] == t_barked
        # Second should be bigram candidate: ran
        assert preds[1] == t_ran
        # Remaining 3 must come from unigram in descending order: fast, cat, night
        assert preds[2] == t_fast
        assert preds[3] == t_cat
        assert preds[4] == t_night

    def test_lstm_masking_all_four_special_tokens(self):
        """Verify LSTM evaluation sets logits to -inf for all 4 special tokens."""
        # Set large positive biases for special tokens, and a moderate positive bias for "dog"
        with torch.no_grad():
            self.model.fc_out.bias.fill_(-100.0)
            for sp_id in self.disallowed:
                self.model.fc_out.bias[sp_id] = 100.0
            # Target token "dog" has higher bias than all other non-special tokens
            self.model.fc_out.bias[self.vocab.token_to_id("dog")] = 50.0

        # Construct test IDs with length >= context_len + 2 so at least 1 index is sampled
        context_len = 5
        test_tokens = [self.vocab.token_to_id("the")] * context_len + [self.vocab.token_to_id("dog"), self.vocab.token_to_id("cat")]
        test_ids = torch.tensor(test_tokens, dtype=torch.long)

        results = evaluate_top_k_accuracy(
            model=self.model,
            baseline=self.baseline,
            test_ids=test_ids,
            vocab=self.vocab,
            seq_len=context_len,
            max_eval_samples=1,
        )

        assert results["evaluation_metadata"]["eligible_samples"] == 1
        # Because special tokens are masked to -inf, "dog" must be Top-1
        assert results["LSTM"]["top1_accuracy"] == 100.0
        assert results["LSTM"]["top5_accuracy"] == 100.0

    def test_identical_eligible_targets_and_denominators(self):
        """Verify LSTM and N-gram use exact same denominator and skip special tokens."""
        context_len = 4
        # Target tokens start at index context_len:
        # Index 4: dog (valid)
        # Index 5: cat (valid)
        # Index 6: unk (excluded)
        # Index 7: eos (excluded)
        # Index 8: pad (excluded)
        # Index 9: bos (excluded)
        # Index 10: barked (valid)
        # Extra padding at end so start=6 is valid
        seq_tokens = [self.vocab.token_to_id("the")] * context_len + [
            self.vocab.token_to_id("dog"),
            self.vocab.token_to_id("cat"),
            self.vocab.unk_id,
            self.vocab.eos_id,
            self.vocab.pad_id,
            self.vocab.bos_id,
            self.vocab.token_to_id("barked"),
            self.vocab.token_to_id("the"),
        ]
        test_ids = torch.tensor(seq_tokens, dtype=torch.long)

        # max_eval_samples=7 so stride=1
        results = evaluate_top_k_accuracy(
            model=self.model,
            baseline=self.baseline,
            test_ids=test_ids,
            vocab=self.vocab,
            seq_len=context_len,
            max_eval_samples=7,
        )

        meta = results["evaluation_metadata"]
        assert meta["total_positions_sampled"] == 7
        assert meta["eligible_samples"] == 3
        assert meta["total_excluded_targets"] == 4
        assert meta["excluded_target_counts"]["<unk>"] == 1
        assert meta["excluded_target_counts"]["<eos>"] == 1
        assert meta["excluded_target_counts"]["<pad>"] == 1
        assert meta["excluded_target_counts"]["<bos>"] == 1

        # Both models must have exact same total_samples
        assert results["LSTM"]["total_samples"] == 3
        assert results["Ngram_Baseline"]["total_samples"] == 3

    def test_zero_eligible_targets_safe_handling(self):
        """Verify evaluate_top_k_accuracy handles 0 eligible targets gracefully without ZeroDivisionError."""
        context_len = 4
        # All tokens in the sequence are <unk>, so all targets are <unk>
        seq_tokens = [self.vocab.unk_id] * 20
        test_ids = torch.tensor(seq_tokens, dtype=torch.long)

        results = evaluate_top_k_accuracy(
            model=self.model,
            baseline=self.baseline,
            test_ids=test_ids,
            vocab=self.vocab,
            seq_len=context_len,
            max_eval_samples=10,
        )

        assert results["evaluation_metadata"]["eligible_samples"] == 0
        assert results["evaluation_metadata"]["total_excluded_targets"] > 0
        assert results["LSTM"]["top1_accuracy"] == 0.0
        assert results["LSTM"]["top3_accuracy"] == 0.0
        assert results["LSTM"]["top5_accuracy"] == 0.0
        assert results["Ngram_Baseline"]["top1_accuracy"] == 0.0
        assert results["Ngram_Baseline"]["top3_accuracy"] == 0.0
        assert results["Ngram_Baseline"]["top5_accuracy"] == 0.0
