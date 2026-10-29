# Model Quality & Implementation Audit Report: SmartKeyboardLSTM

**Audit Date:** 2026-10-27  
**Auditor:** Senior ML Engineer & NLP Researcher  
**System Under Audit:** SmartKeyboardLSTM (`D:\project\SmartKeyboardLSTM`)  
**Base Checkpoint:** `models/best_model.pt` (Epoch 2, Val Loss: 6.0402, Val PPL: 419.96)

---

## 1. Summary of Confirmed Defects, Potential Issues, and Passed Checks

| # | Investigation Area | Status | Findings |
| :--- | :--- | :--- | :--- |
| **1** | **Target Alignment** | **PASSED** | `LMSequenceDataset` correctly aligns $X = \text{tokens}[i : i + L]$ and $Y = \text{tokens}[i + 1 : i + L + 1]$. In `train.py`, `logits[:, t, :]` correctly predicts target `y[:, t]`. |
| **2** | **Preceding Context Length** | **PASSED** | 20 tokens of context are passed during training and truncated appropriately from user prompts during inference. |
| **3** | **Article Boundaries & EOS** | **PASSED WITH OBSERVATION** | `<eos>` is inserted at line breaks in the raw corpus. 17,556 `<eos>` tokens exist in the training set (0.86% of all tokens). |
| **4** | **Vocabulary & Checkpoint IDs** | **PASSED** | Vocabulary size (20,000), special token indices (`<pad>`: 0, `<unk>`: 1, `<eos>`: 2, `<bos>`: 3) match checkpoint metadata exactly. |
| **5** | **Padding & Special Tokens** | **DEFECT IDENTIFIED** | Special token masking was inconsistent between evaluation and inference. See Section 2.1. |
| **6** | **Evaluation Target Context** | **PASSED** | Evaluation uses strictly preceding $L=20$ tokens from the held-out test split to predict token $L+1$. |
| **7** | **LSTM Hidden State Reset** | **PASSED WITH OBSERVATION** | Sequences are shuffled independently per batch, with zero initial hidden state per batch. Within the 20-token BPTT window, recurrence operates correctly. |
| **8** | **Beam Search Candidate States** | **PASSED** | Each beam candidate tracks its own $(h_t, c_t)$ hidden state tuple, cloned across branching expansions. |
| **9** | **Baseline Fairness & Leakage** | **DEFECT IDENTIFIED** | N-gram baseline used training data only, but evaluation permitted the baseline to predict `<unk>`, while forbidding the LSTM from predicting `<unk>`. See Section 2.2. |
| **10**| **Test Sampling Reproducibility** | **PASSED** | Test positions are generated via deterministic strided slicing `range(...)[:1500]`, fully reproducible. |
| **11**| **Perplexity Calculation** | **PASSED** | Test perplexity is calculated from token cross-entropy loss divided by non-padding target tokens ($\exp(\mathcal{L})$). |
| **12**| **OOV & Special Token Masking** | **DEFECT IDENTIFIED** | Asymmetric handling of `<unk>` targets penalized LSTM accuracy by ~4.4%. See Section 2.2. |
| **13**| **Detokenization & Spacing** | **DEFECT IDENTIFIED** | Quotation marks `"` and periods had inconsistent spacing (`" .`), and closing quotes were not bound to punctuation. See Section 2.3. |
| **14**| **Inference Pipeline Parity** | **PASSED** | Both the Streamlit app and evaluation scripts import the identical `AutocompletePredictor` and `BeamSearchDecoder` modules. |

---

## 2. In-Depth Analysis of Identified Defects

### 2.1 Severe Underfitting (Only 2 Epochs on Subsampled Sequences)
- **Root Cause:** Due to initial conservative CPU timing expectations, `train_stride` was increased to 80, and training was stopped after only 2 epochs. The model processed only ~50,800 sequences total (out of 101,658 available sequences per epoch).
- **Consequence:** Training loss remained high at **6.3084** and validation perplexity at **419.96**. At this high loss, the model learned global word/punctuation frequencies (predicting `"the"`, `","`, `"."`, `'"'`), but lacked the semantic capacity to predict domain-appropriate continuations (e.g., following `"released in"` with months or dates).
- **Remedy:** Resume training from `best_model.pt` for 4–8 additional epochs with a smaller sequence stride (`stride = 25-30`) and learning rate decay (`ReduceLROnPlateau`).

### 2.2 Evaluation Disparity: Asymmetric `<unk>` Masking
- **Root Cause:** In `src/evaluate.py`:
  ```python
  # LSTM prediction:
  for sp in [vocab.pad_id, vocab.unk_id, vocab.bos_id]:
      step_logits[sp] = -float("inf")
  top5_lstm = torch.topk(step_logits, k=5).indices.tolist()

  # Baseline prediction:
  top5_ngram = baseline.predict_top_k(context, k=5)
  ```
- **Disparity:** 89 out of 1,500 test positions (5.93%) have target token `<unk>`. The N-gram baseline predicted `<unk>` in its top-5 and received credit **66 times (4.40% of all evaluations)**. The LSTM was explicitly forbidden from predicting `<unk>`, guaranteeing 0% accuracy on all 89 positions.
- **Remedy:** When measuring vocabulary prediction accuracy, either evaluate exclusively on vocabulary words (excluding `<unk>` ground-truth targets) or mask special tokens symmetrically across both models.

### 2.3 Detokenization Formatting
- **Root Cause:** In `Tokenizer.detokenize()`, quotation marks `"` were not categorized in `no_pre_space_punct` or `no_post_space_punct`, causing space insertion before periods (e.g., `" .`).
- **Remedy:** Update regex and detokenization logic to bind quotation marks to adjacent punctuation and avoid stray spaces.

### 2.4 Beam Search Premature Termination & Short Sentence Bias
- **Root Cause:** Beam search allowed 1-token fragments to terminate on `<eos>` or punctuation without enforcing a minimum generation length (`min_length`). With simple power normalization, short fragments (1–2 tokens) achieved higher scores than natural 8–10 token clauses.
- **Remedy:** Implement Google NMT length normalization penalty $\text{lp}(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$, enforce `min_length >= 3` for sentence completions, and disallow immediate termination on punctuation.

---

## 3. Improvement Strategy

1. **Phase 2:** Reproduce baseline measurements under current checkpoint to establish verified reference numbers.
2. **Phase 3:** Resume training from `best_model.pt` for up to 6–8 epochs using adaptive learning rate scheduling and stride 30, saving checkpoints as `models/best_model_v2.pt`.
3. **Phase 4:** Upgrade `BeamSearchDecoder` with Google NMT length penalty, minimum generation length, and enhanced detokenizer spacing.
4. **Phase 5:** Compare original vs. improved models side by side on the identical held-out test set.
5. **Phase 6:** Run regression testing across all 35+ automated tests and verify live Streamlit integration.
