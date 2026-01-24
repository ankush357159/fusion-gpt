import torch
from torch.utils.data import DataLoader
import argparse

from transformers import GPT2TokenizerFast

from src.model.model import GPTConfig
from src.model.pico_gpt import PicoGPT
from src.model.load_gpt2_weights import load_gpt2_weights_into_picogpt
from data.raw import TextDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)  # GPT-2 is larger
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("⚡ Fine-tuning PicoGPT initialized from GPT-2 weights")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    dataset = TextDataset(tokens, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = GPTConfig(
        vocab_size=50257,
        block_size=args.block_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        use_rope=False,  # GPT-2 uses learned positional embeddings
        gpt2_compatible=True,
    )

    model = PicoGPT(config).to(device)

    # Load GPT-2 pretrained weights
    load_gpt2_weights_into_picogpt(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "checkpoints/picogpt_gpt2_finetuned.pt")
    print("GPT-2 fine-tuning complete.")


if __name__ == "__main__":
    main()
