class GPTConfig:
    def __init__(
        self,
        vocab_size,
        block_size=128,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        use_rope=True,
        gpt2_compatible=False,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_rope = use_rope
        self.gpt2_compatible = gpt2_compatible
