class GPTConfig:
    def __init__(
        self,
        vocab_size,
        block_size=128,  # training context length
        max_position_embeddings=1024,  # desired inference length
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.rope_theta = rope_theta
