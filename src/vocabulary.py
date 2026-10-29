"""Vocabulary module for SmartKeyboardLSTM.

Builds, serializes, and manages the token-to-ID and ID-to-token mappings.
Ensures vocabulary is constructed exclusively from the training dataset.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


class Vocabulary:
    """Manages token-to-index mappings and special tokens."""

    def __init__(
        self,
        token_to_id: Optional[Dict[str, int]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.bos_token = bos_token

        self.special_tokens = [pad_token, unk_token, eos_token, bos_token]

        if token_to_id is not None:
            self._token_to_id = dict(token_to_id)
            self._id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}
        else:
            self._token_to_id: Dict[str, int] = {}
            self._id_to_token: Dict[int, str] = {}
            for idx, token in enumerate(self.special_tokens):
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token

        self._frequencies: Dict[str, int] = {}

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.unk_token]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.eos_token]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def frequencies(self) -> Dict[str, int]:
        return self._frequencies

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def token_to_id(self, token: str) -> int:
        """Returns the ID for a token, or the unknown token ID if not present."""
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        """Returns the token for a given ID, or the unknown token if invalid."""
        return self._id_to_token.get(idx, self.unk_token)

    def encode(
        self,
        tokens: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encodes an iterable of tokens into a list of integer token IDs."""
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for t in tokens:
            ids.append(self.token_to_id(t))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        skip_special: bool = True,
    ) -> List[str]:
        """Decodes an iterable of token IDs back into string tokens."""
        special_ids = {self.pad_id, self.unk_id, self.eos_id, self.bos_id}
        tokens: List[str] = []
        for idx in ids:
            if skip_special and idx in special_ids:
                continue
            tokens.append(self.id_to_token(idx))
        return tokens

    @classmethod
    def build_from_tokens(
        cls,
        token_stream: Iterable[str],
        max_vocab_size: int = 20000,
        min_freq: int = 2,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
    ) -> "Vocabulary":
        """Builds a vocabulary exclusively from an iterable of training tokens.
        
        Ranks tokens by frequency, discarding tokens below min_freq,
        and capping at max_vocab_size (including special tokens).
        """
        counter = Counter(token_stream)
        vocab = cls(
            pad_token=pad_token,
            unk_token=unk_token,
            eos_token=eos_token,
            bos_token=bos_token,
        )

        # Reserve slots for special tokens
        slots_remaining = max_vocab_size - len(vocab.special_tokens)

        # Sort tokens by frequency descending, then alphabetically for determinism
        sorted_candidates = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )

        for token, count in sorted_candidates:
            if slots_remaining <= 0:
                break
            if count < min_freq:
                break
            if token not in vocab._token_to_id:
                idx = len(vocab._token_to_id)
                vocab._token_to_id[token] = idx
                vocab._id_to_token[idx] = token
                vocab._frequencies[token] = count
                slots_remaining -= 1

        return vocab

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes vocabulary and configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "token_to_id": self._token_to_id,
            "frequencies": self._frequencies,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Loads a saved vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        vocab = cls(
            token_to_id=payload["token_to_id"],
            pad_token=payload.get("pad_token", "<pad>"),
            unk_token=payload.get("unk_token", "<unk>"),
            eos_token=payload.get("eos_token", "<eos>"),
            bos_token=payload.get("bos_token", "<bos>"),
        )
        vocab._frequencies = payload.get("frequencies", {})
        return vocab
