import torch

# This function is not used in PicoGPT as rope was not a part of GPT-2

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, seq_len, head_dim, device, base_theta, scale_factor):
    # Adjust theta using scaling
    theta = base_theta * scale_factor

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()

    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
