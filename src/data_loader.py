"""Data loading, caching, dataset inspection, and PyTorch dataset creation.

Loads WikiText-2 (wikitext-2-raw-v1) from Hugging Face or cached raw text,
tokenizes each split independently without leakage, builds the vocabulary
exclusively on the training split, and provides memory-efficient PyTorch
Datasets and DataLoaders.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    HF_CACHE_DIR,
    ModelConfig,
    TOKENIZER_CONFIG_FILE,
    VOCAB_FILE,
)
from src.preprocessing import Tokenizer, TokenizerConfig
from src.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LMSequenceDataset(Dataset):
    """Memory-efficient PyTorch Dataset for word-level Language Modeling.
    
    Holds token IDs in a single contiguous 1D tensor.
    Yields input-target pairs as views without materializing duplicate copies.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 20, stride: Optional[int] = None):
        """
        Args:
            token_ids: 1D torch.LongTensor containing all token IDs for the split.
            seq_len: Number of tokens per input sequence.
            stride: Step size between sequence starts. If None, defaults to seq_len (non-overlapping).
        """
        super().__init__()
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be a 1D tensor, got shape {token_ids.shape}")
        
        self.data = token_ids
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Total number of full sequences that can be extracted
        if len(self.data) > self.seq_len:
            self.num_samples = (len(self.data) - 1 - self.seq_len) // self.stride + 1
        else:
            self.num_samples = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x, y


def load_raw_wikitext2(cache_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """Loads WikiText-2 raw v1 lines, caching raw splits to disk on D:.
    
    Uses canonical Hugging Face 'Salesforce/wikitext' repository.
    """
    cache_path = cache_dir or HF_CACHE_DIR
    logger.info("Loading WikiText-2 (wikitext-2-raw-v1) via datasets...")
    
    # Salesforce/wikitext is the canonical repo containing wikitext-2-raw-v1
    hf_dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        cache_dir=str(cache_path),
    )
    
    splits = {
        "train": hf_dataset["train"]["text"],
        "validation": hf_dataset["validation"]["text"],
        "test": hf_dataset["test"]["text"],
    }
    
    # Save raw text files to D:\project\SmartKeyboardLSTM\data\raw for auditability
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, lines in splits.items():
        raw_file = DATA_RAW_DIR / f"{split_name}.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
        logger.info(f"Saved raw {split_name} split ({len(lines)} lines) to {raw_file}")
        
    return splits


def prepare_datasets_and_vocab(
    config: Optional[ModelConfig] = None,
    force_rebuild: bool = False,
) -> Tuple[Vocabulary, Tokenizer, Dict[str, LMSequenceDataset]]:
    """Loads text, creates/loads tokenizer & vocabulary, and creates LMSequenceDatasets.
    
    Guarantees:
    - Vocabulary is fitted EXCLUSIVELY on the training split.
    - Tokenizer is shared and saved.
    - Memory usage is tightly controlled using single 1D tensor representations.
    """
    cfg = config or ModelConfig()
    tokenizer = Tokenizer()
    
    processed_train_pt = DATA_PROCESSED_DIR / "train_ids.pt"
    processed_val_pt = DATA_PROCESSED_DIR / "val_ids.pt"
    processed_test_pt = DATA_PROCESSED_DIR / "test_ids.pt"
    summary_file = DATA_PROCESSED_DIR / "dataset_summary.json"

    # Check if pre-tokenized tensors and vocab already exist and rebuild is not forced
    if (
        not force_rebuild
        and VOCAB_FILE.exists()
        and processed_train_pt.exists()
        and processed_val_pt.exists()
        and processed_test_pt.exists()
    ):
        logger.info(f"Loading existing vocabulary from {VOCAB_FILE}...")
        vocab = Vocabulary.load(VOCAB_FILE)
        
        logger.info("Loading pre-tokenized tensor splits...")
        train_ids = torch.load(processed_train_pt, weights_only=True)
        val_ids = torch.load(processed_val_pt, weights_only=True)
        test_ids = torch.load(processed_test_pt, weights_only=True)
    else:
        # Load raw splits
        splits = load_raw_wikitext2()

        # Step 1: Tokenize train split first to build vocabulary
        logger.info("Tokenizing training split to build vocabulary...")
        train_tokens: List[str] = []
        for line in splits["train"]:
            toks = tokenizer.tokenize(line)
            if toks:
                train_tokens.extend(toks)
                train_tokens.append(cfg.eos_token)

        logger.info(f"Total training tokens collected: {len(train_tokens):,}")
        
        # Step 2: Fit vocabulary ONLY on training tokens
        logger.info(
            f"Fitting vocabulary (max_vocab_size={cfg.max_vocab_size}, min_freq={cfg.min_freq})..."
        )
        vocab = Vocabulary.build_from_tokens(
            token_stream=train_tokens,
            max_vocab_size=cfg.max_vocab_size,
            min_freq=cfg.min_freq,
            pad_token=cfg.pad_token,
            unk_token=cfg.unk_token,
            eos_token=cfg.eos_token,
            bos_token=cfg.bos_token,
        )
        logger.info(f"Built vocabulary with {len(vocab):,} unique tokens.")
        
        # Save vocabulary and tokenizer config
        vocab.save(VOCAB_FILE)
        tokenizer.save_config(TOKENIZER_CONFIG_FILE)
        logger.info(f"Saved vocabulary to {VOCAB_FILE} and tokenizer config to {TOKENIZER_CONFIG_FILE}")

        # Step 3: Convert training tokens to contiguous 1D tensor
        train_ids = torch.tensor(vocab.encode(train_tokens), dtype=torch.long)
        del train_tokens  # Free memory immediately

        # Step 4: Tokenize validation split
        logger.info("Tokenizing validation split...")
        val_tokens: List[str] = []
        for line in splits["validation"]:
            toks = tokenizer.tokenize(line)
            if toks:
                val_tokens.extend(toks)
                val_tokens.append(cfg.eos_token)
        val_ids = torch.tensor(vocab.encode(val_tokens), dtype=torch.long)
        del val_tokens

        # Step 5: Tokenize test split
        logger.info("Tokenizing test split...")
        test_tokens: List[str] = []
        for line in splits["test"]:
            toks = tokenizer.tokenize(line)
            if toks:
                test_tokens.extend(toks)
                test_tokens.append(cfg.eos_token)
        test_ids = torch.tensor(vocab.encode(test_tokens), dtype=torch.long)
        del test_tokens

        # Save processed token tensors for instant resumability
        torch.save(train_ids, processed_train_pt)
        torch.save(val_ids, processed_val_pt)
        torch.save(test_ids, processed_test_pt)

        # Record dataset metadata summary
        summary = {
            "dataset_name": "wikitext-2-raw-v1",
            "source": "Salesforce/wikitext (Hugging Face)",
            "vocab_size": len(vocab),
            "pad_id": vocab.pad_id,
            "unk_id": vocab.unk_id,
            "eos_id": vocab.eos_id,
            "bos_id": vocab.bos_id,
            "train_tokens_count": len(train_ids),
            "val_tokens_count": len(val_ids),
            "test_tokens_count": len(test_ids),
            "seq_len": cfg.seq_len,
            "train_oov_rate": (train_ids == vocab.unk_id).float().mean().item(),
            "val_oov_rate": (val_ids == vocab.unk_id).float().mean().item(),
            "test_oov_rate": (test_ids == vocab.unk_id).float().mean().item(),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved dataset summary to {summary_file}")

    # Build sequence datasets
    # For training, use stride = cfg.train_stride to span the entire corpus efficiently
    train_stride = getattr(cfg, "train_stride", cfg.seq_len)
    datasets = {
        "train": LMSequenceDataset(train_ids, seq_len=cfg.seq_len, stride=train_stride),
        "validation": LMSequenceDataset(val_ids, seq_len=cfg.seq_len, stride=cfg.seq_len),
        "test": LMSequenceDataset(test_ids, seq_len=cfg.seq_len, stride=cfg.seq_len),
    }

    logger.info(
        f"Datasets ready: train={len(datasets['train']):,} batches of seq_len {cfg.seq_len}, "
        f"val={len(datasets['validation']):,}, test={len(datasets['test']):,}"
    )

    return vocab, tokenizer, datasets


def create_dataloaders(
    datasets: Dict[str, LMSequenceDataset],
    batch_size: int = 64,
    eval_batch_size: int = 64,
) -> Dict[str, DataLoader]:
    """Creates efficient PyTorch DataLoaders."""
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }


if __name__ == "__main__":
    vocab, tokenizer, datasets = prepare_datasets_and_vocab(force_rebuild=True)
    dls = create_dataloaders(datasets, batch_size=64)
    x, y = next(iter(dls["train"]))
    print(f"Sample Train Batch: X shape = {x.shape}, Y shape = {y.shape}")
    print(f"X[0, :5] = {x[0, :5]}")
    print(f"Y[0, :5] = {y[0, :5]}")
    print(f"Decoded X[0, :5] = {vocab.decode(x[0, :5].tolist(), skip_special=False)}")
    print(f"Decoded Y[0, :5] = {vocab.decode(y[0, :5].tolist(), skip_special=False)}")
