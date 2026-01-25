from src.model.model import GPTConfig
from src.model.pico_gpt import PicoGPT
from src.model.load_gpt2_weights import load_gpt2_weights_into_picogpt

config = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    use_rope=False,  # GPT-2 uses learned positions
    gpt2_compatible=True,
)

model = PicoGPT(config)
load_gpt2_weights_into_picogpt(model)
