# Text generation after training
import torch


def generate(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(idx[0].tolist())
