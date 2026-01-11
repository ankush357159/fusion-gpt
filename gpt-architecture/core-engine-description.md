## ChatGPT Core Engine Architecture

[Core Engine Architecture](core-engine.mmd)

### High-Level Flow – `Text → Next Token → Repeat`

### 1. Input Text

Raw user text enters the system as a plain string.

### 2. Tokenizer / BPE Encoder

The text is converted into a sequence of **Token IDs** (integers).

Common tokenization methods:

* Byte Pair Encoding (BPE)
* SentencePiece
* WordPiece
* tiktoken

**Output:**
`[token_1, token_2, ..., token_n]`

### 3. Token Embedding Layer

Each token ID is mapped to a dense vector.

Unlike older GPT-style models with learned positional embeddings, modern architectures use **RoPE (Rotary Positional Embeddings)** inside attention layers instead of explicit positional embedding vectors.

**Output:**
Hidden states with shape:
`[batch, sequence_length, hidden_dim]`

### 4. Transformer Stack (N Decoder Layers)

Each layer follows this structure:

```
Input
  ↓
RMSNorm / LayerNorm ①
  ↓
Grouped-Query Attention (GQA)
  ↓
Residual Connection ①
  ↓
RMSNorm / LayerNorm ②
  ↓
SwiGLU / GELU Feed-Forward Network
  ↓
Residual Connection ②
  ↓
Output → Next Layer
```

### 5. Grouped-Query Attention (GQA) – Internal Flow

Inside each attention block:

```
Hidden States
  ↓
Project to Q, K, V
  ↓
Apply RoPE to Q & K
  ↓
Compute Attention Scores (causal masked)
  ↓
KV Cache (reuse past K/V for speed)
  ↓
Weighted Sum of Values
  ↓
Attention Output
```

**Key properties:**

* **RoPE** injects positional information
* **Causal masking** ensures autoregressive behavior
* **KV Cache** avoids recomputing past tokens
* **GQA** reduces memory vs full MHA

### 6. Final Normalization

After all N layers:

```
Final RMSNorm / LayerNorm
```

Stabilizes output before projection to vocabulary space.

### 7. LM Head / Unembedding

A linear projection maps hidden states to vocabulary size:

```
hidden_dim → vocab_size
```

Usually **weight-tied** with the input embedding matrix.

**Output:**
Logits with shape:
`[batch, seq_len, vocab_size]`

### 8. Sampling Pipeline (Inference-Time Controls)

The logits pass through configurable sampling steps:

```
Raw logits
  ↓
Temperature Scaling
  ↓
Top-K Filtering (optional)
  ↓
Top-P / Nucleus Sampling (optional)
  ↓
Repetition Penalty (optional)
  ↓
Frequency / Presence Penalty (optional)
  ↓
Softmax → Probabilities
  ↓
Sample Next Token
```

Each step controls randomness, diversity, and repetition.

### 9. Stopping Criteria

Generation stops when any condition is met:

* Maximum token limit reached
* EOS (End-of-Sequence) token generated
* Stop sequence detected
* Custom termination logic

If none are met:

```
Append token → Re-tokenize → Continue loop
```

### 10. Detokenizer / BPE Decoder

Final token IDs are converted back into readable text.

## Important Optimizations

| Optimization         | Purpose                        | Impact                 |
| -------------------- | ------------------------------ | ---------------------- |
| **KV Cache**         | Reuses past K/V tensors        | 5–40× faster inference |
| **Causal Masking**   | Prevents future token access   | Enables autoregression |
| **GQA**              | Fewer key/value heads          | Lower memory           |
| **RoPE**             | Position encoding in attention | Better long-context    |
| **Dynamic Batching** | Merges live requests           | Higher throughput      |
| **PagedAttention**   | Virtual memory for KV cache    | Memory efficiency      |

## Summary

```
Text
→ Tokenization
→ Token Embeddings
→ ×N [
     RMSNorm
     → GQA (RoPE + KV Cache)
     → Residual
     → RMSNorm
     → SwiGLU FFN
     → Residual
   ]
→ Final RMSNorm
→ LM Head → logits
→ (temperature → top-k/p → penalties)
→ sample next token
→ append & repeat
→ stop → detokenize → output text
```