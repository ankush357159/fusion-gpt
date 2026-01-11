## Training a Mini Language Model from Scratch

Model to learn data from books.

### Pipeline Architecture

```
Books → Cleaning → Tokenization (tiktoken) → Dataset Creation → 
Model Architecture Design → Training from Scratch → Evaluation → 
Inference → Deployment
```

1. **Books** – Collect legally usable book content in digital text format.
2. **Cleaning** – Remove noise like headers, footnotes, and formatting artifacts.
3. **Tokenization** – Convert cleaned text into tokens using tiktoken with GPT-2 vocabulary.
4. **Dataset Creation** – Create training sequences for next-token prediction with train/val splits.
5. **Model Architecture Design** – Define transformer architecture (layers, dimensions, heads).
6. **Training from Scratch** – Train the model using gradient descent and backpropagation.
7. **Evaluation** – Measure loss, perplexity, and text quality on validation set.
8. **Inference** – Generate text by predicting tokens autoregressively from prompts.
9. **Deployment** – Expose the model via API or application interface.

### Key Steps & Details

**1. Tokenization (Using tiktoken + GPT-2 Vocabulary)**
- Use `tiktoken` library with GPT-2 vocabulary: `tiktoken.get_encoding("gpt2")`
- Vocabulary size: **50,257 tokens** (pre-defined)
- Handles subword tokenization automatically via BPE
- Compatible with GPT-2 architecture patterns

**2. Model Architecture Design**

Before training, parameters to decide:
- **Number of layers**: 6-12 (GPT-2 small has 12, GPT-3 has 96)
- **Hidden dimension**: 512-768 (GPT-2 small uses 768, GPT-3 uses 12,288)
- **Attention heads**: 8-12 (must divide hidden dimension evenly)
- **Context window**: 512-1024 tokens (GPT-2 uses 1024)
- **Vocabulary size**: **50,257** (from GPT-2 tokenizer)
- **Total parameters**: Aim for 40M-125M for a "mini" model
  - *Note: ~26M parameters for embeddings (50,257 vocab × 512 dim)*

**3. Dataset Creation Specifics**
- Create training/validation splits (e.g., 90/10 or 95/5)
- Generate sequences of fixed length equal to context window
- Implement next-token prediction: input `[tokens[:-1]]` → target `[tokens[1:]]`
- Tokenize all books using tiktoken: `enc.encode(cleaned_text)`
- Shuffle sequences to prevent overfitting to book order
- Consider overlapping windows to maximize data usage

**4. Training Considerations**

```
Key hyperparameters to define:
- Learning rate: 6e-4 (GPT-2 standard) or 3e-4 to 1e-3 for smaller models
- Batch size: 8-32 sequences (depends on GPU memory and context length)
- Number of epochs: 10-50 (monitor validation loss for early stopping)
- Warmup steps: 2000 (gradual learning rate increase)
- Weight decay: 0.1 (L2 regularization)
- Gradient clipping: 1.0 (prevents exploding gradients)
- Optimizer: AdamW (Adam with decoupled weight decay)
- Learning rate schedule: Cosine decay with warmup
```

### Realistic Expectations for a Mini Model

**Pros:**
- Deep understanding of transformer internals and training dynamics
- Fast iteration cycles for experimentation
- Manageable on consumer GPUs (single GPU with 8-16GB VRAM)
- Learn complete end-to-end training pipeline
- GPT-2 vocabulary is well-tested and efficient

**Cons:**
- Limited capabilities compared to pre-trained models
- May need 10GB+ of quality text data for decent results
- Won't match ChatGPT's performance (needs billions of parameters + RLHF)
- May struggle with rare words or specialized domains
- Requires significant compute time (days of training)

### Recommended Tech Stack

- **Framework**: PyTorch
- **Tokenization**: tiktoken library (`get_encoding("gpt2")`)
- **Monitoring**: Weights & Biases (wandb) or TensorBoard
- **Training**: Single GPU (RTX 3090/4090 with 16-24GB VRAM, or cloud GPU)
- **Books needed**: 20-50 books minimum, ideally 100+ for better results
- **Data volume**: Target 10-50GB of cleaned text

### Suggested Mini Model Specs for Learning

```
Option 1: Ultra-Mini (Fast Training)
- Vocabulary: 50,257 tokens (GPT-2)
- Layers: 6
- Hidden size: 512
- Attention heads: 8
- Context length: 512
- Total parameters: ~40M
  - Token embeddings: ~26M
  - Position embeddings: ~0.26M
  - Transformer blocks: ~14M
- Training time: 1-2 days on single GPU
- GPU memory needed: 6-8GB

Option 2: Mini-GPT-2 (Better Quality)
- Vocabulary: 50,257 tokens (GPT-2)
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Context length: 1024
- Total parameters: ~124M (similar to GPT-2 small)
  - Token embeddings: ~39M
  - Position embeddings: ~0.79M
  - Transformer blocks: ~85M
- Training time: 3-5 days on single GPU
- GPU memory needed: 12-16GB
```

### Additional Recommendations

**For GPT-2 Vocabulary (50,257 tokens):**
- Smaller vocab = fewer embedding parameters = more compute for transformer layers
- Better efficiency compared to 100k vocabulary
- Well-suited for English text (your books)
- Use hidden dimensions divisible by number of heads: 512 (8 heads), 768 (12 heads)

**Training Tips:**
- Start with Option 1 to validate pipeline, then scale to Option 2
- Save checkpoints every 1000-5000 steps
- Generate sample text during training to qualitatively assess progress
- Monitor gradient norms to detect training instabilities
- Use mixed precision training (FP16) to reduce memory and speed up training