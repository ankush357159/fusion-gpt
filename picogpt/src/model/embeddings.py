import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.use_rope = config.use_rope
        if not self.use_rope:
            # GPT-2 style learned positions
            self.pos_emb = nn.Embedding(config.block_size, config.embed_dim)

    def forward(self, idx):
        tok = self.token_emb(idx)

        if self.use_rope:
            return self.dropout(tok)

        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        pos = self.pos_emb(pos)[None, :, :]
        return self.dropout(tok + pos)
