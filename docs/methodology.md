# Methodology & Experimental Protocol: SmartKeyboardLSTM

This document specifies the dataset processing, tokenization, training regimen, baseline construction, and evaluation methodology.

---

## 1. Dataset & Split Integrity

The primary training and evaluation corpus is **WikiText-2** using configuration `wikitext-2-raw-v1`.

### 1.1 Source & Hugging Face Hub Resolution
- **Canonical Repository:** `Salesforce/wikitext` (configuration `wikitext-2-raw-v1`).
- **Technical Resolution:** In modern Hugging Face Hub (`huggingface_hub >= 2.0`), un-namespaced repository IDs like `"wikitext"` are deprecated and rejected with `HfUriError`. The official canonical repository namespace created by Stephen Merity et al. at Salesforce Research (`Salesforce/wikitext`) was identified, downloaded, and verified.
- **Local Storage:** All raw datasets, processed tensors, and Hugging Face cache files are stored exclusively on the **D:** drive under `D:\project\SmartKeyboardLSTM\.cache\huggingface` and `data/raw/`.

### 1.2 Dataset Split Verification
| Split | Raw Line Count | Token Count | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | 36,718 | 2,033,163 | Fitting vocabulary, training model parameters |
| **Validation** | 3,760 | 212,233 | Hyperparameter monitoring, early stopping, best checkpoint selection |
| **Test** | 4,358 | 238,814 | Strictly held-out benchmark evaluation |

**Strict Data Isolation Guarantee:**
- The vocabulary is fitted **exclusively on the training split**. No tokens or frequency statistics from validation or test data are accessible during vocabulary creation.
- Model training gradients are computed **solely on the training split**. Validation and test splits are strictly held out.

---

## 2. Text Preprocessing & Deterministic Tokenization

A single, shared tokenizer (`src/preprocessing.py`) guarantees consistency across training and live inference.

### 2.1 Preprocessing Pipeline
1. **Wiki Header Removal:** Pure section headers (e.g., `= = Early Life = =`) are detected using regex and excluded from sequence modeling to prevent syntactic corruption.
2. **Hyphen Normalization:** WikiText raw text encodes hyphenated words with `@-@` (e.g. `role @-@ playing`). These are normalized to standard hyphens `-`.
3. **Punctuation Isolation:** Standard punctuation marks (`. , ! ? ; : " ( ) [ ] { } -`) are separated with whitespace to preserve their syntactic identity as independent vocabulary tokens.
4. **Lowercasing:** All tokens are converted to lowercase, keeping vocabulary compact while maximizing word prediction coverage.
5. **Whitespace Collapsing:** Redundant consecutive spaces, tabs, and line breaks are normalized to single spaces.

### 2.2 Vocabulary Construction
- The vocabulary contains the **top 20,000 tokens** ranked by frequency from the training set with minimum frequency $\ge 2$.
- Four reserved special tokens are assigned fixed indices:
  - `<pad>`: ID `0` (used for sequence padding, ignored in loss calculation)
  - `<unk>`: ID `1` (assigned to any word outside the 20,000 vocabulary)
  - `<eos>`: ID `2` (marks sequence completion)
  - `<bos>`: ID `3` (marks beginning of text stream)
- Out-of-Vocabulary (OOV) rates under this configuration:
  - Training OOV: **3.92%**
  - Validation OOV: **6.10%**
  - Held-out Test OOV: **7.03%**

---

## 3. Memory-Efficient Batching (`LMSequenceDataset`)

To avoid out-of-memory errors and excessive RAM consumption on local hardware:
1. All token IDs for each split are loaded as a single contiguous 1D `torch.LongTensor` (e.g., training split is an 8 MB tensor).
2. The PyTorch Dataset (`LMSequenceDataset`) computes non-materialized tensor slices on demand:
   ```python
   x = self.data[start : start + seq_len]
   y = self.data[start + 1 : start + seq_len + 1]
   ```
3. PyTorch tensor slicing creates zero-copy memory views. No duplicate copies of sliding windows are held in memory.
4. During training, a sequence stride of 80 tokens spans all 2,033,163 tokens across the entire corpus, generating 25,415 diverse sequences (397 batches of size 64) per epoch.

---

## 4. Training Protocol

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Adaptive momentum for stable recurrent training |
| **Learning Rate** | $1 \times 10^{-3}$ | Standard stable default for LSTM LM |
| **Weight Decay** | $1 \times 10^{-5}$ | $L_2$ regularization on recurrent and projection weights |
| **Batch Size** | 64 | Efficient vectorization on 12-core CPU |
| **Sequence Length** | 20 | Balances context history and BPTT memory |
| **Gradient Clipping** | 1.0 (max norm) | Prevents exploding gradients in recurrent backpropagation |
| **Loss Function** | CrossEntropyLoss | `ignore_index=0` (`<pad>` tokens excluded from loss) |
| **Validation Cadence**| Every 250 batches | Ensures best checkpoint is captured immediately |
| **Early Stopping** | Patience = 3 | Prevents overfitting when validation loss ceases improving |

---

## 5. Baseline Model: N-gram Backoff Model

To establish a competitive baseline, a classical statistical language model was trained on the **identical 2,033,163 training tokens**:
- **Trigram Frequency Table:** Counts trigram tuples $(w_{t-2}, w_{t-1}, w_t)$ (538,505 distinct contexts).
- **Bigram Frequency Table:** Counts bigram pairs $(w_{t-1}, w_t)$ (19,998 distinct contexts).
- **Unigram Frequency Table:** Global token frequencies.
- **Backoff Prediction:** When predicting the next token given context:
  1. If $(w_{t-2}, w_{t-1})$ exists in the trigram table, select the top-$k$ most frequent continuations.
  2. If fewer than $k$ candidates exist, back off to bigram $(w_{t-1})$.
  3. If still fewer than $k$, back off to global unigram frequencies.

---

## 6. Evaluation Protocol

All models are evaluated on the **held-out test split** across 1,500 uniformly sampled positions:
- **Top-1 Accuracy:** Percentage of positions where the target word matches the top predicted candidate.
- **Top-3 Accuracy:** Percentage of positions where the target word appears in the top-3 predictions.
- **Top-5 Accuracy:** Percentage of positions where the target word appears in the top-5 predictions.
- **Test Perplexity:** Exponent of cross-entropy loss over held-out test tokens:
  $$\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^N \mathcal{L}_i\right)$$
- **Prediction Latency:** Measured using high-resolution timers (`time.perf_counter()`) across repeated inferences.
