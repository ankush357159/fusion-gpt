## Training time

### Assumptions

- Model: **GPT-2 (124M parameters)**
- Data: **100 MB cleaned text**
- Tokens: ~15–20 million
- Training type: **Fine-tuning (not from scratch)**
- Epochs: 1–3
- GPU: **T4 / L4 / A100**

### RAM (System Memory)

| Task                           | RAM         |
| ------------------------------ | ----------- |
| Dataset loading + tokenization | 4–6 GB      |
| Training + PyTorch             | 6–8 GB      |
| Safe margin                    | **8–12 GB** |

### Colab

- Free: ~12 GB → **Sufficient**
- Pro / Pro+: More than enough

### GPU VRAM

GPT-2 is small compared to modern LLMs.

| Setup                  | VRAM       |
| ---------------------- | ---------- |
| GPT-2 FP16 fine-tuning | **4–6 GB** |

Colab GPUs:

- T4 (16 GB) → ✔
- L4 (24 GB) → ✔
- A100 (40 GB) → ✔

### Training Time for 100 MB

| GPU  | Time (1–3 epochs) |
| ---- | ----------------- |
| T4   | 1–2 hours         |
| L4   | 45–90 minutes     |
| A100 | 30–60 minutes     |

Tokenization adds ~10–15 minutes.

### Disk Space

| Item        | Size    |
| ----------- | ------- |
| Dataset     | 100 MB  |
| GPT-2 model | ~500 MB |
| Checkpoints | ~1–2 GB |

Total: **~3 GB**

Colab provides ~100 GB.

### Summary

| Resource | Required | Colab |
| -------- | -------- | ----- |
| RAM      | 8–12 GB  | ✔     |
| VRAM     | 4–6 GB   | ✔     |
| Disk     | 3 GB     | ✔     |
| Time     | 1–2 hrs  | ✔     |

### Practical Notes

- GPT-2 uses **BPE tokenizer** (tiktoken-compatible)
- 100 MB is enough for **style/domain adaptation**
- Overfitting risk increases beyond 3 epochs
- Checkpointing every 30 minutes is sufficient

### Final Recommendation

For this setup:

- Colab Free is enough
- 1–2 epochs recommended
- Batch size: 4–8
- Sequence length: 512–1024
