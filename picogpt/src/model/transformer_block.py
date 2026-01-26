from .multi_head_self_attention import MultiHeadSelfAttention
from .feed_forward import FeedForward
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config.embed_dim, config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.resid_dropout(self.attn(self.ln1(x)))
        x = x + self.resid_dropout(self.ff(self.ln2(x)))
        return x
