"""Evaluation pipeline and baseline benchmark for SmartKeyboardLSTM.

Implements:
- N-gram (Trigram/Bigram/Unigram backoff) frequency baseline
- Rigorous held-out test split evaluation
- Top-1, Top-3, and Top-5 next-word accuracy calculation
- Perplexity (PPL) calculation
- Inference latency benchmarking (autocomplete vs beam search)
- Greedy vs Beam Search comparison
- Automated plotting and export of JSON/CSV metrics
"""

from collections import Counter, defaultdict
import json
import logging
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ACCURACY_VS_K_PLOT,
    BASELINE_FILE,
    BEST_MODEL_FILE,
    DATA_PROCESSED_DIR,
    EVAL_RESULTS_CSV,
    EVAL_RESULTS_FAIR_JSON,
    EVAL_RESULTS_JSON,
    MODEL_COMPARISON_PLOT,
    ModelConfig,
    VOCAB_FILE,
)
from src.beam_search import BeamSearchDecoder
from src.model import LSTMNeuralLanguageModel
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NgramBaseline:
    """N-gram frequency baseline with Trigram -> Bigram -> Unigram backoff."""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_tokens = 0

    def fit(self, token_ids: List[int]) -> None:
        """Trains N-gram frequency tables exclusively on training tokens."""
        logger.info(f"Fitting N-gram baseline on {len(token_ids):,} tokens...")
        self.total_tokens = len(token_ids)
        self.unigram_counts.update(token_ids)

        for i in range(len(token_ids) - 1):
            w1 = token_ids[i]
            w2 = token_ids[i + 1]
            self.bigram_counts[w1][w2] += 1

            if i < len(token_ids) - 2:
                w3 = token_ids[i + 2]
                self.trigram_counts[(w1, w2)][w3] += 1

        logger.info(
            f"N-gram fitted: {len(self.unigram_counts):,} unigrams, "
            f"{len(self.bigram_counts):,} bigram contexts, "
            f"{len(self.trigram_counts):,} trigram contexts."
        )

    def predict_top_k(
        self,
        context_ids: List[int],
        k: int = 5,
        disallowed_ids: Optional[Union[Set[int], List[int]]] = None,
    ) -> List[int]:
        """Predicts top-k next tokens using backoff with candidate filtering.

        Never returns disallowed special tokens. Supplements higher-order n-gram
        candidates with unique valid candidates from lower-order backoff levels until
        k valid candidates are collected or all candidate sources are exhausted.
        """
        if disallowed_ids is None:
            disallowed = {
                self.vocab.pad_id,
                self.vocab.unk_id,
                self.vocab.bos_id,
                self.vocab.eos_id,
            }
        else:
            disallowed = set(disallowed_ids)

        candidates: List[int] = []
        seen: Set[int] = set(disallowed)

        # 1. Trigram attempt if context has >= 2 tokens
        if len(context_ids) >= 2:
            key = (context_ids[-2], context_ids[-1])
            if key in self.trigram_counts:
                for tid, _ in self.trigram_counts[key].most_common():
                    if tid not in seen:
                        candidates.append(tid)
                        seen.add(tid)
                        if len(candidates) >= k:
                            return candidates

        # 2. Bigram attempt if context has >= 1 token and still need candidates
        if len(context_ids) >= 1 and len(candidates) < k:
            key1 = context_ids[-1]
            if key1 in self.bigram_counts:
                for tid, _ in self.bigram_counts[key1].most_common():
                    if tid not in seen:
                        candidates.append(tid)
                        seen.add(tid)
                        if len(candidates) >= k:
                            return candidates

        # 3. Unigram fallback / supplement if still need candidates
        if len(candidates) < k:
            for tid, _ in self.unigram_counts.most_common():
                if tid not in seen:
                    candidates.append(tid)
                    seen.add(tid)
                    if len(candidates) >= k:
                        return candidates

        return candidates

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "unigram_counts": self.unigram_counts,
                    "bigram_counts": self.bigram_counts,
                    "trigram_counts": self.trigram_counts,
                    "total_tokens": self.total_tokens,
                },
                f,
            )
        logger.info(f"N-gram baseline saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path, vocab: Vocabulary) -> "NgramBaseline":
        baseline = cls(vocab)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        baseline.unigram_counts = data["unigram_counts"]
        baseline.bigram_counts = data["bigram_counts"]
        baseline.trigram_counts = data["trigram_counts"]
        baseline.total_tokens = data["total_tokens"]
        return baseline


def evaluate_top_k_accuracy(
    model: nn.Module,
    baseline: NgramBaseline,
    test_ids: torch.Tensor,
    vocab: Vocabulary,
    seq_len: int = 20,
    max_eval_samples: int = 1500,
    device: str = "cpu",
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluates Top-1, Top-3, and Top-5 accuracy on the held-out test split under a shared fair policy."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    stride = max(1, (len(test_ids) - seq_len - 1) // max_eval_samples)
    indices = list(range(0, len(test_ids) - seq_len - 1, stride))[:max_eval_samples]

    lstm_correct = {1: 0, 3: 0, 5: 0}
    ngram_correct = {1: 0, 3: 0, 5: 0}
    total_evaluated = 0

    # Disallowed special tokens for candidate predictions and evaluation targets
    special_token_map = {
        vocab.pad_token: vocab.pad_id,
        vocab.unk_token: vocab.unk_id,
        vocab.bos_token: vocab.bos_id,
        vocab.eos_token: vocab.eos_id,
    }
    disallowed_ids = set(special_token_map.values())
    excluded_target_counts = {tok_name: 0 for tok_name in special_token_map}

    lstm_latencies: List[float] = []

    logger.info(
        f"Evaluating Top-K accuracy on {len(indices):,} sampled held-out test positions "
        f"(excluding special token targets: {list(special_token_map.keys())})..."
    )

    with torch.no_grad():
        for start in indices:
            context = test_ids[start : start + seq_len].tolist()
            target = test_ids[start + seq_len].item()

            # Skip test targets that are special tokens, including <unk>
            if target in disallowed_ids:
                target_token = vocab.id_to_token(target)
                excluded_target_counts[target_token] = excluded_target_counts.get(target_token, 0) + 1
                continue

            # --- LSTM Prediction ---
            x_tensor = torch.tensor([context], dtype=torch.long, device=dev)
            t0 = time.perf_counter()
            logits, _ = model(x_tensor)
            step_logits = logits[0, -1, :].clone()

            # Mask out all four special tokens (<pad>, <unk>, <bos>, <eos>)
            for sp in disallowed_ids:
                if 0 <= sp < len(step_logits):
                    step_logits[sp] = -float("inf")
            top5_lstm = torch.topk(step_logits, k=5).indices.tolist()
            lstm_latencies.append((time.perf_counter() - t0) * 1000.0)

            # --- N-Gram Prediction (with shared disallowed tokens and backoff) ---
            top5_ngram = baseline.predict_top_k(context, k=5, disallowed_ids=disallowed_ids)

            # Check accuracies on the exact same eligible position
            for k in [1, 3, 5]:
                if target in top5_lstm[:k]:
                    lstm_correct[k] += 1
                if target in top5_ngram[:k]:
                    ngram_correct[k] += 1

            total_evaluated += 1

    total_excluded = sum(excluded_target_counts.values())
    logger.info(
        f"Evaluation finished: {total_evaluated:,} eligible positions evaluated, "
        f"{total_excluded:,} special token targets excluded {excluded_target_counts}."
    )

    safe_denom = total_evaluated if total_evaluated > 0 else 1.0

    results = {
        "LSTM": {
            "top1_accuracy": round((lstm_correct[1] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "top3_accuracy": round((lstm_correct[3] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "top5_accuracy": round((lstm_correct[5] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "avg_latency_ms": round(float(np.mean(lstm_latencies)), 2) if lstm_latencies else 0.0,
            "total_samples": total_evaluated,
        },
        "Ngram_Baseline": {
            "top1_accuracy": round((ngram_correct[1] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "top3_accuracy": round((ngram_correct[3] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "top5_accuracy": round((ngram_correct[5] / safe_denom) * 100, 2) if total_evaluated > 0 else 0.0,
            "total_samples": total_evaluated,
        },
        "evaluation_metadata": {
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else str(BEST_MODEL_FILE),
            "allowed_token_policy": "Excluded <pad>, <unk>, <bos>, <eos> from candidate predictions and evaluation targets",
            "disallowed_tokens": list(special_token_map.keys()),
            "total_positions_sampled": len(indices),
            "eligible_samples": total_evaluated,
            "total_excluded_targets": total_excluded,
            "excluded_target_counts": excluded_target_counts,
        },
    }
    return results


def evaluate_beam_vs_greedy(
    decoder: BeamSearchDecoder,
    test_prompts: List[str],
) -> List[Dict]:
    """Compares sequence generation using Greedy decoding vs Beam Search."""
    comparisons = []
    logger.info("Comparing Greedy vs Beam Search decoding...")

    for prompt in test_prompts:
        t_greedy_0 = time.perf_counter()
        greedy_cand = decoder.decode_greedy(prompt, max_length=10)
        t_greedy = (time.perf_counter() - t_greedy_0) * 1000.0

        t_beam_0 = time.perf_counter()
        beam_cands = decoder.decode(prompt, beam_width=3, max_length=10)
        t_beam = (time.perf_counter() - t_beam_0) * 1000.0

        comparisons.append(
            {
                "prompt": prompt,
                "greedy_text": greedy_cand.text,
                "greedy_log_prob": round(greedy_cand.cumulative_log_prob, 3),
                "greedy_latency_ms": round(t_greedy, 2),
                "beam_candidates": [
                    {
                        "text": c.text,
                        "log_prob": round(c.cumulative_log_prob, 3),
                        "score": round(c.normalized_score, 3),
                    }
                    for c in beam_cands
                ],
                "beam_latency_ms": round(t_beam, 2),
            }
        )
    return comparisons


def plot_comparison_metrics(metrics: Dict, output_path: Path) -> None:
    """Generates a professional bar plot comparing LSTM vs N-gram baseline."""
    labels = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"]
    lstm_vals = [
        metrics["LSTM"]["top1_accuracy"],
        metrics["LSTM"]["top3_accuracy"],
        metrics["LSTM"]["top5_accuracy"],
    ]
    ngram_vals = [
        metrics["Ngram_Baseline"]["top1_accuracy"],
        metrics["Ngram_Baseline"]["top3_accuracy"],
        metrics["Ngram_Baseline"]["top5_accuracy"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, lstm_vals, width, label="LSTM Model", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, ngram_vals, width, label="N-gram Baseline", color="#ff7f0e")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Next-Word Prediction Accuracy (Held-out Test Split)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved model comparison plot to {output_path}")


def run_full_evaluation(
    model_path: Optional[Path] = None,
    max_eval_samples: int = 1500,
    output_json_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    output_plot_path: Optional[Path] = None,
) -> Dict:
    """Executes the full evaluation pipeline and exports all metrics and plots."""
    ckpt_path = model_path or BEST_MODEL_FILE
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Train the model first.")

    vocab = Vocabulary.load(VOCAB_FILE)
    tokenizer = Tokenizer()

    # Load test IDs
    test_ids = torch.load(DATA_PROCESSED_DIR / "test_ids.pt", weights_only=True)
    train_ids = torch.load(DATA_PROCESSED_DIR / "train_ids.pt", weights_only=True)

    # Load or fit N-gram baseline
    if BASELINE_FILE.exists():
        baseline = NgramBaseline.load(BASELINE_FILE, vocab)
    else:
        baseline = NgramBaseline(vocab)
        baseline.fit(train_ids.tolist())
        baseline.save(BASELINE_FILE)

    # Load LSTM model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("config", {})
    model = LSTMNeuralLanguageModel(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.2),
        pad_idx=vocab.pad_id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Top-K accuracy under shared fair policy
    accuracy_results = evaluate_top_k_accuracy(
        model=model,
        baseline=baseline,
        test_ids=test_ids,
        vocab=vocab,
        seq_len=20,
        max_eval_samples=max_eval_samples,
        checkpoint_path=ckpt_path,
    )

    # Beam vs Greedy comparison
    decoder = BeamSearchDecoder(model=model, vocab=vocab, tokenizer=tokenizer)
    prompts = [
        "The game was released in",
        "During the Second World War ,",
        "The city is located in the",
        "It was one of the most",
    ]
    beam_comparisons = evaluate_beam_vs_greedy(decoder, prompts)

    # Calculate test perplexity across contiguous test sequences
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    test_loss_sum = 0.0
    test_tokens_sum = 0
    eval_seq_len = 20
    eval_batch_size = 64
    num_seqs = (len(test_ids) - 1) // eval_seq_len
    with torch.no_grad():
        for b in range(0, num_seqs, eval_batch_size):
            b_end = min(b + eval_batch_size, num_seqs)
            x_list = [test_ids[i * eval_seq_len : (i + 1) * eval_seq_len] for i in range(b, b_end)]
            y_list = [test_ids[i * eval_seq_len + 1 : (i + 1) * eval_seq_len + 1] for i in range(b, b_end)]
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)
            logits, _ = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            non_pad = (y_batch != vocab.pad_id).sum().item()
            test_loss_sum += loss.item() * non_pad
            test_tokens_sum += non_pad

    test_loss = test_loss_sum / max(test_tokens_sum, 1)
    test_ppl = math.exp(min(test_loss, 20.0))

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": str(ckpt_path),
        "val_loss": checkpoint.get("val_loss", None),
        "val_ppl": checkpoint.get("val_ppl", None),
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(test_ppl, 2),
        "accuracy_metrics": accuracy_results,
        "beam_vs_greedy_samples": beam_comparisons,
    }

    # Save to JSON
    target_json = output_json_path or EVAL_RESULTS_JSON
    target_json.parent.mkdir(parents=True, exist_ok=True)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved evaluation JSON to {target_json}")

    # Save to CSV
    csv_rows = [
        {"Model": "LSTM", "Metric": "Top-1 Accuracy (%)", "Value": accuracy_results["LSTM"]["top1_accuracy"]},
        {"Model": "LSTM", "Metric": "Top-3 Accuracy (%)", "Value": accuracy_results["LSTM"]["top3_accuracy"]},
        {"Model": "LSTM", "Metric": "Top-5 Accuracy (%)", "Value": accuracy_results["LSTM"]["top5_accuracy"]},
        {"Model": "LSTM", "Metric": "Avg Autocomplete Latency (ms)", "Value": accuracy_results["LSTM"]["avg_latency_ms"]},
        {"Model": "LSTM", "Metric": "Test Perplexity", "Value": round(test_ppl, 2)},
        {"Model": "Ngram_Baseline", "Metric": "Top-1 Accuracy (%)", "Value": accuracy_results["Ngram_Baseline"]["top1_accuracy"]},
        {"Model": "Ngram_Baseline", "Metric": "Top-3 Accuracy (%)", "Value": accuracy_results["Ngram_Baseline"]["top3_accuracy"]},
        {"Model": "Ngram_Baseline", "Metric": "Top-5 Accuracy (%)", "Value": accuracy_results["Ngram_Baseline"]["top5_accuracy"]},
    ]
    df = pd.DataFrame(csv_rows)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved evaluation CSV to {output_csv_path}")
    elif output_json_path is None:
        # Default run writes to standard EVAL_RESULTS_CSV
        df.to_csv(EVAL_RESULTS_CSV, index=False)
        logger.info(f"Saved evaluation CSV to {EVAL_RESULTS_CSV}")

    # Plot model comparison
    target_plot = output_plot_path or (MODEL_COMPARISON_PLOT if output_json_path is None else None)
    if target_plot is not None:
        plot_comparison_metrics(accuracy_results, target_plot)

    return final_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SmartKeyboardLSTM models.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--max-samples", type=int, default=1500, help="Maximum evaluation samples")
    parser.add_argument("--output-json", type=str, default=None, help="Output path for JSON report")
    parser.add_argument("--output-csv", type=str, default=None, help="Output path for CSV metrics")
    parser.add_argument("--output-plot", type=str, default=None, help="Output path for comparison plot")
    args = parser.parse_args()

    out_json = Path(args.output_json) if args.output_json else None
    out_csv = Path(args.output_csv) if args.output_csv else None
    out_plot = Path(args.output_plot) if args.output_plot else None
    m_path = Path(args.model_path) if args.model_path else None

    run_full_evaluation(
        model_path=m_path,
        max_eval_samples=args.max_samples,
        output_json_path=out_json,
        output_csv_path=out_csv,
        output_plot_path=out_plot,
    )
