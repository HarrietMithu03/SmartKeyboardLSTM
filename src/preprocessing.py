"""Text preprocessing, tokenization, and detokenization module.

Provides a unified, deterministic tokenization pipeline shared by
both training and inference. Handles text normalization, punctuation,
contractions, and WikiText-specific formatting.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import List, Union


@dataclass
class TokenizerConfig:
    """Configuration for text tokenization."""
    lowercase: bool = True
    clean_wiki_headers: bool = True
    replace_wiki_hyphens: bool = True
    split_punctuation: bool = True


class Tokenizer:
    """Deterministic word-level tokenizer and detokenizer."""

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        
        # Regex to isolate punctuation while keeping contractions intact
        # WikiText uses separated punctuation frequently, but user input might not.
        self._punct_pattern = re.compile(r"([.,!?;:\"()\[\]{}–—])")
        self._wiki_header_pattern = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")

    def clean_line(self, line: str) -> str:
        """Cleans a single line of text according to the configuration."""
        if not line:
            return ""

        text = line.strip()
        if not text:
            return ""

        # Clean WikiText headers: lines like '= = Heading = ='
        if self.config.clean_wiki_headers:
            header_match = self._wiki_header_pattern.match(text)
            if header_match:
                # Discard pure wiki section headers to avoid cluttering syntax
                return ""

        # Replace WikiText hyphen marker '@-@' with standard '-'
        if self.config.replace_wiki_hyphens:
            text = text.replace("@-@", "-")
            text = text.replace("@,@ ", ", ")
            text = text.replace("@.@ ", ". ")

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Separate punctuation with spaces for word-level tokenization
        if self.config.split_punctuation:
            text = self._punct_pattern.sub(r" \1 ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes an input string into a list of word/punctuation tokens."""
        cleaned = self.clean_line(text)
        if not cleaned:
            return []
        return cleaned.split()

    def detokenize(self, tokens: List[str], filter_special: bool = True) -> str:
        """Reconstructs natural readable text from a list of tokens.
        
        Properly handles punctuation spacing, contractions, and special tokens.
        """
        if not tokens:
            return ""

        special_tokens = {"<pad>", "<unk>", "<eos>", "<bos>"}
        filtered = []
        for t in tokens:
            if filter_special and t in special_tokens:
                continue
            filtered.append(t)

        if not filtered:
            return ""

        no_pre_space_punct = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'", "%"}
        no_post_space_punct = {"(", "[", "{"}
        contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "n't"}

        result: List[str] = []
        in_quote = False

        for i, token in enumerate(filtered):
            if i == 0:
                if token == '"':
                    in_quote = True
                result.append(token)
                continue

            prev_token = filtered[i - 1]

            if token == '"':
                if in_quote:
                    result.append(token)
                    in_quote = False
                else:
                    result.append(" " + token)
                    in_quote = True
            elif token in no_pre_space_punct or token in contractions:
                result.append(token)
            elif prev_token in no_post_space_punct:
                result.append(token)
            elif prev_token == '"' and in_quote:
                result.append(token)
            elif token == "-" and prev_token.isalnum():
                result.append(token)
            elif prev_token == "-" and token.isalnum():
                result.append(token)
            else:
                result.append(" " + token)

        text = "".join(result).strip()
        if text.endswith("-"):
            text = text[:-1].rstrip()
        return text

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Saves tokenizer configuration to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> "Tokenizer":
        """Loads a tokenizer from a saved configuration JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(TokenizerConfig(**config_dict))
