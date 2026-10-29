"""Training pipeline for LSTM Language Model.

Supports:
- Model instantiation and weight initialization
- Memory-efficient batch iteration
- Cross-entropy loss with padding ignore
- Gradient clipping
- Validation tracking and Perplexity (PPL) calculation
- Best model and resumable checkpoint saving
- Early stopping
- Training curves plotting and metric persistence
- Smoke test mode for immediate pre-flight validation
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import (
    BEST_MODEL_FILE,
    LAST_CHECKPOINT_FILE,
    ModelConfig,
    PLOTS_DIR,
    TRAIN_HISTORY_FILE,
    TRAINING_CURVES_PLOT,
    VOCAB_FILE,
)
from src.data_loader import create_dataloaders, prepare_datasets_and_vocab
from src.model import LSTMNeuralLanguageModel
from src.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducible experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluates the model on a dataloader.
    
    Returns:
        avg_loss: Average Cross-Entropy loss per token
        ppl: Perplexity = exp(avg_loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # x shape: (B, T), y shape: (B, T)
            logits, _ = model(x)
            # Flatten to calculate token-level loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Non-padding token count
            non_pad = (y != criterion.ignore_index).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))  # Cap to avoid math overflow
    return avg_loss, ppl


def save_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_ppl: float,
    train_loss: float,
    config: ModelConfig,
    vocab: Vocabulary,
) -> None:
    """Saves complete model checkpoint with metadata."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "train_loss": train_loss,
        "vocab_size": len(vocab),
        "pad_idx": vocab.pad_id,
        "unk_idx": vocab.unk_id,
        "eos_idx": vocab.eos_id,
        "bos_idx": vocab.bos_id,
        "config": {
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "seq_len": config.seq_len,
        },
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath} (Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f})")


def plot_training_curves(history: List[Dict], output_path: Path) -> None:
    """Generates and saves professional training/validation loss and perplexity plots."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_ppls = [h["val_ppl"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "s-", color="#ff7f0e", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Perplexity curve
    ax2.plot(epochs, val_ppls, "d-", color="#2ca02c", label="Validation Perplexity", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Perplexity (PPL)", fontsize=12)
    ax2.set_title("Validation Perplexity Over Epochs", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved training curves plot to {output_path}")


def train_model(
    config: Optional[ModelConfig] = None,
    smoke_test: bool = False,
    resume: bool = False,
    resume_from: Optional[Union[str, Path]] = None,
    save_best_as: Optional[Union[str, Path]] = None,
    save_last_as: Optional[Union[str, Path]] = None,
    history_file: Optional[Union[str, Path]] = None,
    curves_plot_file: Optional[Union[str, Path]] = None,
) -> Tuple[LSTMNeuralLanguageModel, Vocabulary, List[Dict]]:
    """Executes the full training loop or a rapid smoke test.
    
    Args:
        config: Hyperparameter configuration
        smoke_test: If True, runs 1 epoch on limited batches to verify functionality
        resume: If True, resumes from default last_checkpoint.pt
        resume_from: Explicit checkpoint path to resume from
        save_best_as: Destination path for the best validation checkpoint
        save_last_as: Destination path for the latest checkpoint
        history_file: Path to save the training history JSON
        curves_plot_file: Path to save the training curves plot
    """
    cfg = config or ModelConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    logger.info(f"Using training device: {device}")

    best_ckpt_target = Path(save_best_as) if save_best_as else BEST_MODEL_FILE
    last_ckpt_target = Path(save_last_as) if save_last_as else LAST_CHECKPOINT_FILE
    hist_target = Path(history_file) if history_file else TRAIN_HISTORY_FILE
    plot_target = Path(curves_plot_file) if curves_plot_file else TRAINING_CURVES_PLOT

    # Prepare datasets and vocabulary
    vocab, tokenizer, datasets = prepare_datasets_and_vocab(cfg)
    dataloaders = create_dataloaders(
        datasets,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
    )

    # Instantiate model
    model = LSTMNeuralLanguageModel(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_idx=vocab.pad_id,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized. Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history: List[Dict] = []
    patience_counter = 0

    # Resume from checkpoint if requested and available
    ckpt_source = None
    if resume_from and Path(resume_from).exists():
        ckpt_source = Path(resume_from)
    elif resume and LAST_CHECKPOINT_FILE.exists():
        ckpt_source = LAST_CHECKPOINT_FILE

    if ckpt_source is not None:
        logger.info(f"Resuming training from {ckpt_source}...")
        ckpt = torch.load(ckpt_source, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if hist_target.exists():
            with open(hist_target, "r", encoding="utf-8") as f:
                history = json.load(f)
        elif TRAIN_HISTORY_FILE.exists():
            with open(TRAIN_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        logger.info(f"Resumed from epoch {start_epoch - 1} with best val loss: {best_val_loss:.4f}")

    if hasattr(os, "cpu_count") and os.cpu_count():
        torch.set_num_threads(os.cpu_count())

    num_epochs = 1 if smoke_test else cfg.num_epochs
    max_train_batches = 30 if smoke_test else cfg.max_train_batches
    max_eval_batches = 15 if smoke_test else None

    total_train_batches = len(dataloaders["train"])
    if max_train_batches is not None:
        total_train_batches = min(total_train_batches, max_train_batches)

    logger.info(
        f"Starting training run: epochs={num_epochs}, batches_per_epoch={total_train_batches}, "
        f"batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, smoke_test={smoke_test}, "
        f"threads={torch.get_num_threads()}"
    )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_tokens = 0
        batch_idx = 0
        last_log_time = epoch_start

        for x, y in dataloaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            non_pad = (y != vocab.pad_id).sum().item()
            running_loss += loss.item() * non_pad
            running_tokens += non_pad
            batch_idx += 1

            # Log every 50 batches
            if batch_idx % 50 == 0 or batch_idx == total_train_batches:
                curr_loss = running_loss / max(running_tokens, 1)
                now = time.time()
                step_rate = 50.0 / max(now - last_log_time, 0.01)
                logger.info(
                    f"Epoch {epoch:2d} | Batch {batch_idx:4d}/{total_train_batches} "
                    f"({batch_idx / total_train_batches * 100:.1f}%) | "
                    f"Loss: {curr_loss:.4f} | {step_rate:.1f} batches/s | "
                    f"Time: {now - epoch_start:.1f}s"
                )
                last_log_time = now

            # Periodic mid-epoch validation every 250 batches
            if not smoke_test and batch_idx % 250 == 0:
                mid_val_loss, mid_val_ppl = evaluate(
                    model, dataloaders["validation"], criterion, device, max_batches=30
                )
                logger.info(
                    f"--> Mid-Epoch {epoch} Eval (Batch {batch_idx}) | "
                    f"Val Loss: {mid_val_loss:.4f} | Val PPL: {mid_val_ppl:.2f}"
                )
                if mid_val_loss < best_val_loss:
                    best_val_loss = mid_val_loss
                    save_checkpoint(
                        best_ckpt_target,
                        model,
                        optimizer,
                        epoch,
                        mid_val_loss,
                        mid_val_ppl,
                        running_loss / max(running_tokens, 1),
                        cfg,
                        vocab,
                    )
                model.train()

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_loss = running_loss / max(running_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))

        # Evaluate on validation set
        val_loss, val_ppl = evaluate(
            model,
            dataloaders["validation"],
            criterion,
            device,
            max_batches=max_eval_batches,
        )
        elapsed = time.time() - epoch_start

        # Step learning rate scheduler
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:2d}/{start_epoch + num_epochs - 1:2d} | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
            f"Val Loss: {val_loss:.4f} (Val PPL: {val_ppl:.2f}) | "
            f"LR: {curr_lr:.6f} | Time: {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "learning_rate": curr_lr,
            "elapsed_seconds": round(elapsed, 2),
        }
        history.append(epoch_record)

        # Save last checkpoint for resumability
        save_checkpoint(
            last_ckpt_target,
            model,
            optimizer,
            epoch,
            val_loss,
            val_ppl,
            train_loss,
            cfg,
            vocab,
        )

        # Retain best validation checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                best_ckpt_target,
                model,
                optimizer,
                epoch,
                val_loss,
                val_ppl,
                train_loss,
                cfg,
                vocab,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience and not smoke_test:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # Persist training history JSON
    with open(hist_target, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    if len(history) > 0 and not smoke_test:
        plot_training_curves(history, plot_target)

    logger.info("Training process completed.")
    return model, vocab, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartKeyboard LSTM Language Model")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 1-epoch smoke test")
    parser.add_argument("--epochs", type=int, default=ModelConfig.num_epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=ModelConfig.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=ModelConfig.learning_rate, help="Learning rate")
    parser.add_argument("--train_stride", type=int, default=40, help="Stride between sequence starts for training")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Max batches per epoch")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Explicit checkpoint to resume from")
    parser.add_argument("--save_best_as", type=str, default=None, help="Path to save best checkpoint")
    parser.add_argument("--save_last_as", type=str, default=None, help="Path to save last checkpoint")
    parser.add_argument("--history_file", type=str, default=None, help="Path to save history JSON")
    parser.add_argument("--curves_plot_file", type=str, default=None, help="Path to save curves plot PNG")
    args = parser.parse_args()

    cfg = ModelConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_stride=args.train_stride,
        max_train_batches=args.max_train_batches,
    )
    train_model(
        cfg,
        smoke_test=args.smoke_test,
        resume=args.resume,
        resume_from=args.resume_from,
        save_best_as=args.save_best_as,
        save_last_as=args.save_last_as,
        history_file=args.history_file,
        curves_plot_file=args.curves_plot_file,
    )
