"""Unit tests for preprocessing, tokenization, detokenization, and vocabulary."""

import json
from pathlib import Path
import pytest

from src.preprocessing import Tokenizer, TokenizerConfig
from src.vocabulary import Vocabulary


class TestTokenizer:
    """Tests for Tokenizer cleaning, tokenization, and detokenization."""

    def setup_method(self):
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        text = "Hello world! This is a test."
        tokens = self.tokenizer.tokenize(text)
        assert tokens == ["hello", "world", "!", "this", "is", "a", "test", "."]

    def test_clean_wiki_headers(self):
        header = "= = History of Science = ="
        cleaned = self.tokenizer.clean_line(header)
        assert cleaned == ""

    def test_clean_wiki_hyphens(self):
        raw = "role @-@ playing game"
        tokens = self.tokenizer.tokenize(raw)
        assert "-" in tokens or "role-playing" in " ".join(tokens)

    def test_empty_and_whitespace_input(self):
        assert self.tokenizer.tokenize("") == []
        assert self.tokenizer.tokenize("   \n\t  ") == []

    def test_detokenize_punctuation_spacing(self):
        tokens = ["the", "city", ",", "located", "in", "france", "."]
        detok = self.tokenizer.detokenize(tokens)
        assert detok == "the city, located in france."

    def test_detokenize_skips_special_tokens(self):
        tokens = ["<bos>", "hello", "<unk>", "world", "<eos>"]
        detok = self.tokenizer.detokenize(tokens, filter_special=True)
        assert detok == "hello world"

    def test_tokenizer_config_persistence(self, tmp_path):
        cfg_file = tmp_path / "tok_cfg.json"
        self.tokenizer.save_config(cfg_file)
        loaded_tok = Tokenizer.load_config(cfg_file)
        assert loaded_tok.config.lowercase == self.tokenizer.config.lowercase
        assert loaded_tok.config.clean_wiki_headers == self.tokenizer.config.clean_wiki_headers


class TestVocabulary:
    """Tests for Vocabulary encoding, decoding, building, and persistence."""

    def setup_method(self):
        self.corpus = ["the", "apple", "is", "on", "the", "table", "the", "apple"]
        self.vocab = Vocabulary.build_from_tokens(
            self.corpus,
            max_vocab_size=10,
            min_freq=1,
        )

    def test_special_tokens_present(self):
        assert self.vocab.pad_token in self.vocab
        assert self.vocab.unk_token in self.vocab
        assert self.vocab.eos_token in self.vocab
        assert self.vocab.bos_token in self.vocab
        assert self.vocab.pad_id == 0
        assert self.vocab.unk_id == 1

    def test_most_frequent_ranked_first(self):
        # 'the' appears 3 times, 'apple' appears 2 times
        the_id = self.vocab.token_to_id("the")
        apple_id = self.vocab.token_to_id("apple")
        assert the_id < apple_id

    def test_unknown_token_handling(self):
        unknown_id = self.vocab.token_to_id("nonexistent_word_xyz")
        assert unknown_id == self.vocab.unk_id

    def test_encode_decode_roundtrip(self):
        input_tokens = ["the", "apple", "table"]
        encoded = self.vocab.encode(input_tokens)
        decoded = self.vocab.decode(encoded, skip_special=True)
        assert decoded == input_tokens

    def test_vocabulary_save_and_load(self, tmp_path):
        save_file = tmp_path / "vocab.json"
        self.vocab.save(save_file)
        loaded = Vocabulary.load(save_file)

        assert len(loaded) == len(self.vocab)
        assert loaded.token_to_id("the") == self.vocab.token_to_id("the")
        assert loaded.pad_id == self.vocab.pad_id
        assert loaded.unk_id == self.vocab.unk_id
