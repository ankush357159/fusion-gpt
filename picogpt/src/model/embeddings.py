import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, idx):
        return self.token_emb(idx)
