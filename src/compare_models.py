"""Model comparison utility: benchmarks v1 vs v2 checkpoints on held-out test split."""

import json
import logging
import math
from pathlib import Path
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    BASELINE_FILE,
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    VOCAB_FILE,
)
from src.beam_search import BeamSearchDecoder
from src.evaluate import NgramBaseline
from src.model import LSTMNeuralLanguageModel
from src.preprocessing import Tokenizer
from src.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_ckpt(ckpt_path: Path, vocab: Vocabulary) -> LSTMNeuralLanguageModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = LSTMNeuralLanguageModel(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_model_on_test(
    model: LSTMNeuralLanguageModel,
    test_ids: torch.Tensor,
    vocab: Vocabulary,
    max_samples: int = 1500,
    seq_len: int = 20,
) -> Dict[str, float]:
    stride = max(1, (len(test_ids) - seq_len - 1) // max_samples)
    indices = list(range(0, len(test_ids) - seq_len - 1, stride))[:max_samples]

    top1 = 0
    top3 = 0
    top5 = 0
    latencies = []
    total = 0

    with torch.no_grad():
        for start in indices:
            context = test_ids[start : start + seq_len].tolist()
            target = test_ids[start + seq_len].item()
            if target == vocab.pad_id:
                continue

            x = torch.tensor([context], dtype=torch.long)
            t0 = time.perf_counter()
            logits, _ = model(x)
            step_logits = logits[0, -1, :].clone()
            for sp in [vocab.pad_id, vocab.unk_id, vocab.bos_id]:
                step_logits[sp] = -float("inf")
            top5_preds = torch.topk(step_logits, k=5).indices.tolist()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if target in top5_preds[:1]:
                top1 += 1
            if target in top5_preds[:3]:
                top3 += 1
            if target in top5_preds[:5]:
                top5 += 1
            total += 1

    return {
        "top1_accuracy": round((top1 / total) * 100, 2),
        "top3_accuracy": round((top3 / total) * 100, 2),
        "top5_accuracy": round((top5 / total) * 100, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "total_samples": total,
    }


def compute_test_perplexity(
    model: LSTMNeuralLanguageModel,
    test_ids: torch.Tensor,
    vocab: Vocabulary,
) -> float:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    test_loss_sum = 0.0
    test_tokens_sum = 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - 21, 20):
            x = test_ids[i : i + 20].unsqueeze(0)
            y = test_ids[i + 1 : i + 21].unsqueeze(0)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            non_pad = (y != vocab.pad_id).sum().item()
            test_loss_sum += loss.item() * non_pad
            test_tokens_sum += non_pad
    avg_loss = test_loss_sum / max(test_tokens_sum, 1)
    return round(math.exp(min(avg_loss, 20.0)), 2)


def run_comparison(
    v1_ckpt_path: Path = MODELS_DIR / "best_model_v1.pt",
    v2_ckpt_path: Path = MODELS_DIR / "best_model_v2.pt",
) -> Dict:
    vocab = Vocabulary.load(VOCAB_FILE)
    tokenizer = Tokenizer()
    test_ids = torch.load(DATA_PROCESSED_DIR / "test_ids.pt", weights_only=True)

    logger.info("Evaluating V1 model checkpoint...")
    v1_model = load_model_from_ckpt(v1_ckpt_path, vocab)
    v1_acc = evaluate_model_on_test(v1_model, test_ids, vocab)
    v1_ppl = compute_test_perplexity(v1_model, test_ids, vocab)

    logger.info("Evaluating V2 model checkpoint...")
    v2_model = load_model_from_ckpt(v2_ckpt_path, vocab)
    v2_acc = evaluate_model_on_test(v2_model, test_ids, vocab)
    v2_ppl = compute_test_perplexity(v2_model, test_ids, vocab)

    prompts = [
        "The game was released in",
        "During the Second World War ,",
        "The city is located in the",
        "It was one of the most",
    ]

    v1_dec = BeamSearchDecoder(v1_model, vocab, tokenizer)
    v2_dec = BeamSearchDecoder(v2_model, vocab, tokenizer)

    qualitative_samples = []
    for p in prompts:
        t0_v1 = time.perf_counter()
        cands_v1 = v1_dec.decode(p, beam_width=3, max_length=10, min_length=3)
        t_v1 = (time.perf_counter() - t0_v1) * 1000.0

        t0_v2 = time.perf_counter()
        cands_v2 = v2_dec.decode(p, beam_width=3, max_length=10, min_length=3)
        t_v2 = (time.perf_counter() - t0_v2) * 1000.0

        qualitative_samples.append({
            "prompt": p,
            "v1_greedy": v1_dec.decode_greedy(p, max_length=10).text,
            "v1_beam_top": cands_v1[0].text if cands_v1 else "",
            "v1_latency_ms": round(t_v1, 2),
            "v2_greedy": v2_dec.decode_greedy(p, max_length=10).text,
            "v2_beam_top": cands_v2[0].text if cands_v2 else "",
            "v2_latency_ms": round(t_v2, 2),
        })

    comparison_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v1": {
            "checkpoint": str(v1_ckpt_path),
            "accuracy": v1_acc,
            "test_perplexity": v1_ppl,
        },
        "v2": {
            "checkpoint": str(v2_ckpt_path),
            "accuracy": v2_acc,
            "test_perplexity": v2_ppl,
        },
        "qualitative_comparison": qualitative_samples,
    }

    out_file = METRICS_DIR / "v1_vs_v2_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved comparison JSON to {out_file}")

    # Plot comparison chart
    labels = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"]
    v1_vals = [v1_acc["top1_accuracy"], v1_acc["top3_accuracy"], v1_acc["top5_accuracy"]]
    v2_vals = [v2_acc["top1_accuracy"], v2_acc["top3_accuracy"], v2_acc["top5_accuracy"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    r1 = ax.bar(x - width / 2, v1_vals, width, label="V1 Model (Epoch 2)", color="#7f7f7f")
    r2 = ax.bar(x + width / 2, v2_vals, width, label="V2 Model (Improved)", color="#2ca02c")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("V1 vs V2 Model Comparison (Held-out Test Split)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_file = PLOTS_DIR / "v1_vs_v2_comparison.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_file}")

    return comparison_results


if __name__ == "__main__":
    run_comparison()
