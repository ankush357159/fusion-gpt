## ChatGPT Core Engine Architecture

[Core Engine Architecture](core-engine.mmd)

### High-Level Flow – `Text → Next Token → Repeat`

1. **Input Text**  
   → Raw string gets passed to the tokenizer

2. **Tokenizer / BPE Encoder**  
   → Converts text into sequence of **Token IDs** (integers)  
   Most common nowadays: Byte-Pair Encoding (BPE), SentencePiece, WordPiece, tiktoken, etc.

3. **Embedding Layer**  
   Two kinds of embeddings are added together:
   - **Token Embeddings** – learned vector for each vocabulary item
   - **Positional Embeddings** – information about token position  
     (learned, RoPE, ALiBi, relative positional bias, etc.)

   → Result: **hidden states** (usually shape `[batch, sequence_length, hidden_dim]`)

4. **Transformer Stack** (N × identical decoder layers)

   Each layer consists of (in the most common order):

   ```
   Input to layer
      ↓
   LayerNorm ①
      ↓
   Multi-Head Causal Self-Attention
      ↓
   Residual connection (+ input)
      ↓
   LayerNorm ②
      ↓
   MLP / Feed-Forward Network   (usually SwiGLU, GeGLU or Gated Linear + small amount of experts in MoE models)
      ↓
   Residual connection (+ attention output)
      ↓
   Output of this layer → input to next layer
   ```

5. **After all N layers**  
   → Final **Layer Normalization** (very important in modern architectures)

6. **LM Head / Unembedding**  
   - Linear layer: hidden_dim → vocab_size  
   - Usually **tied** with input embedding matrix (weight sharing)  
   → Produces **logits** [batch, seq_len, vocab_size]

7. **Sampling & Generation Controls** (during inference)

   Typical chain of operations (order can slightly vary):

   ```
   raw logits
      ↓
   Temperature scaling          (1.0 = default, >1 more random, <1 more deterministic)
      ↓
   Optional: Top-K filtering
      ↓
   Optional: Top-P (nucleus) sampling
      ↓
   Optional: Repetition penalty
      ↓
   Optional: Frequency / Presence penalty
      ↓
   Softmax → probabilities
      ↓
   Sample next token (multinomial / greedy / beam / etc.)
   ```

8. **Stopping Criteria** (any of these usually stop generation)

   - Reached **max_new_tokens** / max context length
   - Generated **EOS** token (most important in practice)
   - Found one of the **stop sequences** (programmed stop strings)
   - Custom logic (safety filter, length ratio, etc.)

9. **Important Optimizations**

   | Optimization          | What it does                                                                 | Huge impact on |
   |-----------------------|------------------------------------------------------------------------------|----------------|
   | **KV Cache**          | Stores previous Key/Value matrices → only compute new token's Q/K/V         | Speed ×5–40×   |
   | **Causal masking**    | Attention can only look left (past tokens)                                   | Autoregressive property |
   | **Batching**          | Process multiple requests at once + padding + attention mask                | Throughput     |
   | **Dynamic batching**  | Continuously add new requests to running batch, evict finished ones         | Real-time serving |
   | **PagedAttention**    | (not shown but very common now) virtual memory style KV cache management    | Memory efficiency |

### Summary

```
Text
→ Tokenization
→ Embed + PosEnc
→ ×N [LN → Causal MHA → residual → LN → MLP → residual]
→ Final LN
→ LM head → logits
→ (temperature → top-k/p → repetition/freq penalty) → sample
→ append & repeat   or   stop + decode → output text
```