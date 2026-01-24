import torch.nn as nn
import torch.nn.functional as F
from .embeddings import GPTEmbeddings
from .transformer_block import TransformerBlock


class PicoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = GPTEmbeddings(config)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
