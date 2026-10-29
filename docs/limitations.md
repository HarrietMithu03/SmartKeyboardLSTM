# Known Limitations and Constraints: SmartKeyboardLSTM

This document candidly assesses the technical boundaries, data characteristics, and operational trade-offs of the current implementation.

---

## 1. Domain Discrepancy (Wikipedia vs. Conversational Text)

### 1.1 Non-Conversational Data Distribution
The system is trained exclusively on **WikiText-2** (`wikitext-2-raw-v1`), a corpus derived from featured articles on Wikipedia.
- **Consequence:** The model excels at formal historical narratives, geography, video game descriptions, and biographical summaries (e.g., predicting *"during the second world war, the army..."*).
- **Limitation:** The vocabulary and probability distribution do **not** reflect informal mobile keyboard communications:
  - No conversational idioms, slang, or colloquial contractions (e.g., *"gonna"*, *"idk"*, *"brb"*, *"omw"*).
  - No emoji or emoji-word co-occurrence modeling.
  - Punctuation heavily emphasizes formal encyclopedic conventions (e.g., quotation marks, citations, parentheses, semicolons) rather than casual messaging patterns.

---

## 2. Vocabulary Coverage & Out-of-Vocabulary (OOV) Rate

### 2.1 Fixed 20,000-Word Vocabulary
- The vocabulary is capped at 20,000 unique word tokens with a minimum occurrence frequency of 2 in the training set.
- On the held-out test split, the Out-of-Vocabulary rate is **7.03%**.
- **Behavior on Unknown Words:** Any word not present in the vocabulary is mapped to `<unk>`. During inference:
  - If a user types an OOV word (e.g., an unfamiliar name or modern technical jargon), the input representation treats that token as `<unk>`.
  - While the model continues predicting valid words using preceding context, it cannot reliably predict the specific OOV token itself.
  - Special tokens like `<unk>` are intentionally masked from the UI suggestion chips to avoid showing cryptic tokens to end-users.

---

## 3. Hardware & Compute Constraints

### 3.1 Local CPU Execution
- In compliance with the environment constraints, the system runs locally on the user's Intel Iris Xe integrated graphics machine using CPU PyTorch across 12 threads.
- **Latency Performance:**
  - **Single next-word autocomplete:** **13.18 ms** (fast and well within the 50 ms budget for 60 FPS real-time typing).
  - **Beam Search (Width 3, 10 tokens):** **160–190 ms** (satisfactory for on-demand user clicks, but too computationally intensive to execute automatically on every keystroke).
- **Scale Trade-offs:** Deeper architectures (e.g., 3-layer LSTMs with 1024 hidden dimensions or Transformer decoders) would exceed real-time interactive thresholds on standard CPU hardware without GPU acceleration.

---

## 4. Context Window Size

### 4.1 20-Token Context Window
- The model operates with a sequence context window of **20 tokens**.
- While 20 words is sufficient for capturing immediate clause structure and local agreement (e.g., subject-verb alignment), long-range document discourse or topical themes across multiple paragraphs are beyond this horizon.

---

## 5. Repetition Tendencies in Autoregressive Generation

### 5.1 Degeneration in Long Horizon Decoding
- Like all standard recurrent language models trained with cross-entropy, extended open-ended generation beyond 15–20 tokens can exhibit repetitive phrases (e.g., *"and the first ... and the first"*).
- The implementation mitigates this via a configurable **repetition penalty (1.15)** and **length normalization ($\alpha = 0.7$)**, but sentence completion is most effective when constrained to 5–12 tokens.
