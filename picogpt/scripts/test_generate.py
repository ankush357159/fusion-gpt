import torch

from configs.GPTConfig import GPTConfig
from src.inference.generate import generate
from src.model.pico_gpt import PicoGPT
from src.tokenizer.tokenizer import TiktokenTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = TiktokenTokenizer(encoding_name="gpt2")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        use_rope=False,
        gpt2_compatible=False,
    )

    model = PicoGPT(config).to(device)
    model.block_size = config.block_size

    checkpoint_path = "checkpoints/picogpt_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    prompt = "Once upon a time"
    output = generate(model, tokenizer, prompt, max_new_tokens=50, device=device)
    print(output)


if __name__ == "__main__":
    main()
