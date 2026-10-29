# System Architecture: SmartKeyboardLSTM

This document details the neural network architecture, data structures, and algorithmic design of the Smart Keyboard autocompletion system.

---

## 1. High-Level Architecture Overview

```
                      +-----------------------------+
                      |     Raw User Prompt /       |
                      |   Text Buffer in Editor     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Deterministic Tokenizer   |
                      |  (Punctuation & Contractions|
                      |    Lowercasing & Cleaning)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Vocabulary Token-to-ID    |
                      | (Vocab size = 20,000 words) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Context Truncation View   |
                      |   (Last <= 20 token IDs)    |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
  +-----------------------+                     +-----------------------+
  |  Feature A: Next-Word |                     | Feature B: Sentence   |
  |  Autocomplete Engine  |                     | Beam Search Decoder   |
  +-----------+-----------+                     +-----------+-----------+
              |                                             |
              | Softmax over top-3                          | Multi-step Beam
              | words (masked specials)                     | Expansion & Ranking
              v                                             v
  +-----------------------+                     +-----------------------+
  | 3 Suggestion Chips    |                     | Top 3 Candidate       |
  | with Probabilities %  |                     | Sentence Continuations|
  +-----------+-----------+                     +-----------+-----------+
              |                                             |
              +----------------------+----------------------+
                                     | Click to Insert
                                     v
                      +-----------------------------+
                      |    Updated Editor Buffer    |
                      +-----------------------------+
```

---

## 2. Neural Language Model (`LSTMNeuralLanguageModel`)

The language model is a word-level recurrent neural network that computes the conditional probability distribution:
$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

### 2.1 Model Components & Tensor Shapes

| Layer | Type | Configuration | Input Tensor Shape | Output Tensor Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding** | `nn.Embedding` | $V = 20{,}000$, $D = 128$, `pad_idx=0` | `(B, T)` | `(B, T, 128)` |
| **Input Dropout** | `nn.Dropout` | $p = 0.20$ | `(B, T, 128)` | `(B, T, 128)` |
| **LSTM Cell** | `nn.LSTM` | `input_size=128`, `hidden_size=256`, `num_layers=1`, `batch_first=True` | `(B, T, 128)` | `(B, T, 256)` |
| **Output Dropout** | `nn.Dropout` | $p = 0.20$ | `(B, T, 256)` | `(B, T, 256)` |
| **Projection Head**| `nn.Linear` | `in_features=256`, `out_features=20000`, `bias=True` | `(B, T, 256)` | `(B, T, 20000)` |

Total Trainable Parameters: **8,095,264**.

### 2.2 Weight Initialization
- **Embedding weights:** Uniform distribution $\mathcal{U}(-0.08, 0.08)$, with zeroed padding vector.
- **LSTM input-to-hidden weights:** Xavier uniform initialization.
- **LSTM hidden-to-hidden weights:** Orthogonal initialization to preserve gradient norm across time steps.
- **LSTM forget gate bias:** Initialized to `1.0` to encourage initial retention of long-term memory.
- **Projection head:** Weights initialized from $\mathcal{U}(-0.08, 0.08)$, biases initialized to `0.0`.

### 2.3 Single-Step Inference Interface (`step`)
For auto-regressive decoding, running full sequence forwards is inefficient. The model provides a dedicated single-step method:
```python
def step(self, token_id: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    # token_id: (B, 1)
    # hidden: ((1, B, 256), (1, B, 256))
    # returns: logits (B, 20000), next_hidden
```
This enables Beam Search to execute constant-time transitions per expansion step ($O(1)$ with respect to sequence history length).

---

## 3. Custom Beam Search Architecture

The `BeamSearchDecoder` maintains a set of $W$ parallel candidate hypotheses, expanding and pruning paths at each step.

### 3.1 Hypothesis State Representation
```python
@dataclass
class BeamHypothesis:
    token_ids: List[int]
    cumulative_log_prob: float
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
    is_finished: bool
```
Crucially, **each hypothesis carries its own isolated hidden state tuple $(h_t, c_t)$**. When a hypothesis branches into multiple children, the parent's $(h_t, c_t)$ tensors are cloned, ensuring no state interference across search paths.

### 3.2 Scoring and Length Normalization Policy
Because probabilities $P(w_t) \le 1$, cumulative log probabilities strictly decrease with sequence length. Without normalization, beam search prematurely favors truncated 1-token fragments.

The decoder employs exponential length normalization:
$$\text{Score}(Y) = \frac{\sum_{t=1}^L \log P(y_t \mid y_{<t}, X)}{L^\alpha}$$
- $\alpha = 0.0$: Pure cumulative log-probability (favors shortest continuations).
- $\alpha = 0.7$ (Default): Balanced normalization rewarding coherent, multi-word phrases.
- $\alpha = 1.0$: Full linear length average.

### 3.3 Greedy Equivalence
When $W = 1$, $\alpha = 0$, and temperature $= 1.0$, `BeamSearchDecoder` mathematically and empirically matches pure greedy decoding:
$$w_t = \arg\max_w P(w \mid w_{<t})$$
This identity is verified in automated unit test `test_beam_width_one_equals_greedy`.

### 3.4 Duplicate Suppression
Different hypothesis paths can occasionally emit identical detokenized textual continuations. The decoder filters out duplicate textual candidate outputs before presentation to the interface.
