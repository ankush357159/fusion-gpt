### How to Run Different Files in Google Collab

Because the project is modular, files should be run as modules, not by double-click style execution.

#### General Rule `python -m package.subpackage.filename`

1. Import function and run

```py
from data.raw.data import load_data

dataset = load_data()
```

2. If file.py have `main()` function then run `!python -m data.raw.data`


### PicoGPT Architecture (Decoder-Only Transformer)
```mathematica
Input Text
   │
   ▼
Tokenizer (tiktoken GPT-2 BPE)
   │
   ▼
Token IDs  ──► shape: (B, T)
   │
   ▼
Token Embedding Layer
(Lookup table: vocab_size × n_embd)
   │
   ▼
Positional Encoding (RoPE)
   │
   ▼
Sum: Token Embedding + Positional Info
   │
   ▼
Dropout
   │
   ▼
┌──────────────────────────────────────┐
│         Transformer Block × N        │
│      (repeated n_layers times)       │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ LayerNorm                     │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Masked Multi-Head Attention   │◄─┼── Causal Mask
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Dropout                       │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Residual Connection (+)       │  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ LayerNorm                     │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Feed Forward Network (MLP)    │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ GELU Activation               │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Linear Projection             │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Dropout                       │  │
│  │   │                           │  │
│  │   ▼                           │  │
│  │ Residual Connection (+)       │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Linear Output Projection (LM Head)
(weights tied with token embeddings)
   │
   ▼
Logits  ──► shape: (B, T, vocab_size)
   │
   ▼
Softmax (during inference only)
   │
   ▼
Next Token Probabilities
```

### Inside Multi-Head Attention
```mathematica
Input (B, T, n_embd)
   │
   ├─► Linear → Q (queries)
   ├─► Linear → K (keys)
   └─► Linear → V (values)

Split into heads → (B, n_heads, T, head_dim)

Attention Scores = (Q · Kᵀ) / √head_dim
        │
        ▼
Add Causal Mask (prevent looking ahead)
        │
        ▼
Softmax
        │
        ▼
Weighted sum with V
        │
        ▼
Concatenate heads
        │
        ▼
Linear Projection → Output
```

### Data Flow During Training
```mathematica
Input Tokens:      [t1, t2, t3, t4]
Target Tokens:     [t2, t3, t4, t5]

Model predicts probability distribution at each position:
Position 1 → predict t2
Position 2 → predict t3
Position 3 → predict t4
Position 4 → predict t5
```