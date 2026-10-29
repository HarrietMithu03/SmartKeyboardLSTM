"""Configuration module for SmartKeyboardLSTM.

Centralizes paths, hyperparameters, and environment settings.
Ensures all storage, data, and checkpoints remain on the D: drive.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import torch


# Root workspace directory on D:
BASE_DIR = Path(r"D:\project\SmartKeyboardLSTM").resolve()

# Directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DOCS_DIR = BASE_DIR / "docs"
CACHE_DIR = BASE_DIR / ".cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"

# Ensure all primary runtime directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, HF_CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Point Hugging Face and PyTorch caches to D:
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")


@dataclass
class ModelConfig:
    """Hyperparameters for the LSTM Language Model."""
    # Vocabulary & Tokens
    max_vocab_size: int = 20000
    min_freq: int = 2
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    eos_token: str = "<eos>"
    bos_token: str = "<bos>"
    
    # Model Architecture
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    
    # Sequence & Training
    seq_len: int = 20
    train_stride: int = 40
    batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_epochs: int = 5
    early_stopping_patience: int = 3
    max_train_batches: Optional[int] = None
    
    # Device & Seed
    device: str = "cpu"
    seed: int = 42


@dataclass
class BeamSearchConfig:
    """Hyperparameters for Beam Search generation."""
    beam_width: int = 3
    max_length: int = 12
    length_penalty_alpha: float = 0.7  # Exponent for length normalization
    temperature: float = 1.0
    repetition_penalty: float = 1.15
    min_length: int = 1
    no_repeat_ngram_size: int = 0  # 0 = disabled, >0 blocks repeating n-grams
    prefix_repetition_window: int = 0  # 0 = disabled, >0 penalizes recent prefix tokens
    clean_unbalanced_quotes: bool = False  # True balances or removes dangling isolated quotes


@dataclass
class AppConfig:
    """Application UI settings."""
    app_title: str = "Smart Keyboard Autocomplete"
    top_k_suggestions: int = 3
    default_beam_width: int = 3
    max_beam_width: int = 8
    default_max_length: int = 10
    max_generation_length: int = 25
    example_prompts: List[str] = field(default_factory=lambda: [
        "The game was released in",
        "During the Second World War ,",
        "The city is located in the",
        "It was one of the most",
        "According to the official report ,",
        "The music was composed by",
        "The first season of the television",
        "After the battle , the army",
    ])


# Artifact Filepaths
VOCAB_FILE = MODELS_DIR / "vocab.json"
TOKENIZER_CONFIG_FILE = MODELS_DIR / "tokenizer_config.json"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pt"
LAST_CHECKPOINT_FILE = MODELS_DIR / "last_checkpoint.pt"
BASELINE_FILE = MODELS_DIR / "ngram_baseline.pkl"

TRAIN_HISTORY_FILE = METRICS_DIR / "training_history.json"
EVAL_RESULTS_JSON = METRICS_DIR / "evaluation_results.json"
EVAL_RESULTS_FAIR_JSON = METRICS_DIR / "evaluation_results_fair.json"
EVAL_RESULTS_CSV = METRICS_DIR / "evaluation_results.csv"
TRAINING_CURVES_PLOT = PLOTS_DIR / "training_curves.png"
MODEL_COMPARISON_PLOT = PLOTS_DIR / "model_comparison.png"
ACCURACY_VS_K_PLOT = PLOTS_DIR / "accuracy_vs_k.png"
LATENCY_PLOT = PLOTS_DIR / "inference_latency.png"
