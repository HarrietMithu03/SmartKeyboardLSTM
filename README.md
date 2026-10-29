# Smart Keyboard: Text Autocompletion Using LSTM and Beam Search

An end-to-end, locally running deep learning application for smart keyboard typing, featuring **Feature A (Real-Time Next-Word Autocomplete)** and **Feature B (Multi-Token Sentence Completion using Custom Beam Search)**.

Trained on the **WikiText-2** benchmark corpus with PyTorch, evaluated on held-out test data against a classical N-gram baseline, and served via an interactive, polished **Streamlit** user interface.

---

## 📸 Key Features

1. **Feature A: Next-Word Autocompletion**
   - Predicts next-word probabilities over a 20,000-word vocabulary in **~13 ms** on CPU.
   - Displays top-3 suggestion chips with confidence percentages.
   - One-click insertion into the editor with intelligent spacing, punctuation, and contraction handling.
   - Filters out internal special tokens (`<pad>`, `<unk>`, `<eos>`, `<bos>`) from end-user chips.

2. **Feature B: Custom Beam Search Sentence Completion**
   - Multi-token exploratory continuation generator using cumulative log-probabilities.
   - Configurable beam width ($W = 1$ to $8$), max length ($3$ to $20$ tokens), and length penalty exponent ($\alpha$).
   - Mathematically verified: $W=1$ matches pure Greedy decoding.
   - Independent per-beam LSTM hidden state preservation across expansion paths.
   - Duplicate candidate suppression and repetition penalty.

3. **Benchmarking & Evaluation Dashboard**
   - Full evaluation on the held-out WikiText-2 test split.
   - Comparison against an N-gram backoff frequency baseline trained on the same data.
   - Top-1, Top-3, Top-5 accuracy benchmarks, test perplexity, and latency profiles.

---

## 📁 Project Directory Structure

```
SmartKeyboardLSTM/
├── app.py                      # Interactive Streamlit application
├── config.py                   # Centralized paths and hyperparameters (all on D:)
├── requirements.txt            # Frozen, verified dependencies
├── README.md                   # Complete guide & documentation
├── .gitignore                  # Git exclusions for models, datasets, venv, caches
├── data/
│   ├── raw/                    # Downloaded raw text splits (train.txt, valid.txt, test.txt)
│   └── processed/              # Tokenized 1D tensors & dataset summary
├── models/
│   ├── best_model.pt           # Best validation checkpoint (Val Loss: 6.0402, Val PPL: 419.96)
│   ├── last_checkpoint.pt      # Resumable training checkpoint
│   ├── vocab.json              # 20,000-word vocabulary mapping
│   ├── tokenizer_config.json   # Tokenizer configuration
│   └── ngram_baseline.pkl      # Trained N-gram frequency baseline
├── outputs/
│   ├── metrics/                # Machine-readable JSON and CSV evaluation metrics
│   │   ├── training_history.json
│   │   ├── evaluation_results.json
│   │   └── evaluation_results.csv
│   └── plots/                  # Generated publication-quality figures
│       ├── training_curves.png
│       └── model_comparison.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # WikiText-2 loading, zero-copy LMSequenceDataset, DataLoaders
│   ├── preprocessing.py        # Tokenizer & detokenizer with punctuation/contraction handling
│   ├── vocabulary.py           # Vocabulary builder (strict train-only fit, no leakage)
│   ├── model.py                # PyTorch LSTMNeuralLanguageModel with single-step inference
│   ├── train.py                # Training loop with validation, clipping, early stopping, plots
│   ├── predict.py              # Next-word autocomplete inference engine and buffer editor
│   ├── beam_search.py          # Custom Beam Search decoder with length normalization
│   └── evaluate.py             # Test split evaluation pipeline, N-gram baseline, metrics exporter
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Tokenization, detokenization, vocabulary roundtrips
│   ├── test_model.py           # Forward shapes, finite outputs, step inference, checkpoints
│   ├── test_beam_search.py     # Beam search constraints, greedy equivalence, scoring
│   ├── test_integration.py     # End-to-end integration across model, decoder, and text editor
│   └── test_e2e_simulation.py  # User workflow simulation and artifact validity verification
└── docs/
    ├── architecture.md         # Network design, tensor shapes, Beam Search state tracking
    ├── methodology.md          # Dataset splitting, tokenization, training protocol, baselines
    ├── results.md              # Empirical metrics, loss curves, comparison tables, generation
    └── limitations.md          # Known constraints, Wikipedia domain mismatch, compute trade-offs
```

---

## 🚀 Quickstart Guide (PowerShell)

### 1. Environment Setup
The project uses an isolated virtual environment at `D:\project\SmartKeyboardLSTM\.venv` with all caches directed to the `D:` drive.

```powershell
# Set environment variables for UTF-8 encoding and isolated D: caches
$env:PYTHONIOENCODING="utf-8"
$env:PIP_CACHE_DIR="D:\project\SmartKeyboardLSTM\.cache\pip"
$env:HF_HOME="D:\project\SmartKeyboardLSTM\.cache\huggingface"

# Activate the virtual environment
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\Activate.ps1"
```

### 2. Download Dataset & Prepare Vocabulary
Downloads `wikitext-2-raw-v1`, builds the 20,000-word vocabulary **strictly on training tokens**, and generates zero-copy 1D sequence datasets:

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\python.exe" -m src.data_loader
```

### 3. Run the Training Smoke Test
Quick 1-epoch smoke test on a small subset of batches to verify forward pass, backward pass, evaluation, and checkpoint saving:

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\python.exe" -m src.train --smoke_test
```

### 4. Train the Model
Trains the LSTM on CPU (leveraging all available cores):

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\python.exe" -m src.train --epochs 2 --train_stride 80
```
- Best validation model saved automatically to: `models/best_model.pt`
- Training curves saved to: `outputs/plots/training_curves.png`
- Epoch metrics saved to: `outputs/metrics/training_history.json`

### 5. Run Held-Out Test Evaluation
Evaluates the model against the N-gram baseline on the held-out test split:

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\python.exe" -m src.evaluate
```
- Results saved to `outputs/metrics/evaluation_results.json` and `outputs/metrics/evaluation_results.csv`
- Comparison figure saved to `outputs/plots/model_comparison.png`

### 6. Launch the Streamlit Web Application
Starts the live typing interface:

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\streamlit.exe" run app.py --server.port 8501
```
Open your browser to: **`http://localhost:8501`**

---

## 🧪 Running Automated Tests

Run the full automated test suite with pytest:

```powershell
& "D:\project\SmartKeyboardLSTM\.venv\Scripts\python.exe" -m pytest -v
```
All **35 automated unit, integration, and end-to-end tests** pass cleanly.

---

## 📊 Summary of Evaluation Results

| Evaluation Metric | PyTorch LSTM Model | N-gram Frequency Baseline |
| :--- | :---: | :---: |
| **Top-1 Next-Word Accuracy** | **13.67%** | 17.73% |
| **Top-3 Next-Word Accuracy** | **23.40%** | 30.27% |
| **Top-5 Next-Word Accuracy** | **28.33%** | 37.13% |
| **Test Perplexity (PPL)** | **399.11** | N/A (zeros on novel n-grams) |
| **Average Prediction Latency** | **13.18 ms** | ~0.4 ms |
| **Beam Search Latency (10 tokens)** | **162.8 ms** | N/A (incoherent generation) |

*For full experimental details, refer to [`docs/results.md`](file:///D:/project/SmartKeyboardLSTM/docs/results.md).*

---

## 🛠️ Troubleshooting

- **Out of Memory (RAM):** The project uses contiguous 1D memory views (`LMSequenceDataset`). If physical memory is restricted, reduce batch size via `--batch_size 32`.
- **Checkpoint Not Found:** Ensure you ran `src.train` prior to launching the Streamlit interface or running integration tests.
- **Port 8501 In Use:** Specify a different port using `--server.port 8502`.
