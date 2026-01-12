## Training time

Assumptions:

- Base model: 7B parameter LLM (e.g., Mistral / LLaMA / Phi-2)
- Method: **QLoRA / LoRA fine-tuning**
- Token count: ~180M
- Colab GPU: **T4 / L4 / A100 (varies by tier)**

### RAM (System Memory)

| Task                              | RAM Needed   |
| --------------------------------- | ------------ |
| Dataset loading + preprocessing   | 8–12 GB      |
| Training (Transformers + PyTorch) | 12–16 GB     |
| Safe margin                       | **16–24 GB** |

### Colab Availability

| Colab Tier | RAM    |
| ---------- | ------ |
| Free       | ~12 GB |
| Pro        | ~25 GB |
| Pro+       | ~32 GB |

**Recommendation:** Colab **Pro or Pro+** is ideal for stability.

### GPU VRAM

| Setup              | VRAM     |
| ------------------ | -------- |
| 7B + QLoRA (4-bit) | 8–12 GB  |
| 7B + LoRA (FP16)   | 16–24 GB |

Colab GPUs:

| GPU  | VRAM  |
| ---- | ----- |
| T4   | 16 GB |
| L4   | 24 GB |
| A100 | 40 GB |

**QLoRA is strongly recommended** for Colab.

### Training Time (1 GB of C4)

| GPU  | Estimated Time |
| ---- | -------------- |
| T4   | 12–20 hours    |
| L4   | 8–12 hours     |
| A100 | 4–6 hours      |

This assumes:

- Batch size: 1–2
- Gradient accumulation
- 1–2 epochs

### Disk Space

| Item             | Space   |
| ---------------- | ------- |
| C4 subset (1 GB) | 1 GB    |
| Base model       | 8–15 GB |
| Checkpoints      | 5–10 GB |

**Total:** ~20–25 GB

Colab provides ~100 GB, so space is not an issue.

### Reality Check for Colab

| Constraint            | Impact                       |
| --------------------- | ---------------------------- |
| 12-hour session limit | Training may get interrupted |
| GPU availability      | A100 not guaranteed          |
| Idle timeout          | Long runs need activity      |

**Best practice:** Save checkpoints to Google Drive every 30–60 minutes.

### Summary

| Resource | Required | Colab Pro          |
| -------- | -------- | ------------------ |
| RAM      | 16–24 GB | ✔                  |
| VRAM     | 8–12 GB  | ✔                  |
| Disk     | 25 GB    | ✔                  |
| Time     | 8–20 hrs | ⚠ (session limits) |

### Strong Recommendation

For 1 GB of C4:

- Use **QLoRA**
- Train for **1 epoch**
- Save checkpoints frequently
- Prefer **Colab Pro+** or split training across sessions

If needed, step-by-step setup can be provided for:

- Loading a 1 GB C4 subset
- Tokenization
- QLoRA config for Colab
- Checkpoint recovery
- Cost-optimized cloud alternatives
