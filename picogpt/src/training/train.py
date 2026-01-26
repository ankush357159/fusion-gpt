import argparse
import math

import torch
from torch.utils.data import DataLoader

from configs.GPTConfig import GPTConfig
from src.model.pico_gpt import PicoGPT
from src.training.dataset import load_and_split_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training PicoGPT from scratch")

    train_dataset, val_dataset, test_dataset = load_and_split_datasets(
        tokens_path=args.tokens_path,
        block_size=args.block_size,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    config = GPTConfig(
        vocab_size=50257,
        block_size=args.block_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        use_rope=False,
        gpt2_compatible=False,
    )

    model = PicoGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(0, args.warmup_steps)

    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if step >= total_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/picogpt_best.pt")
            print("Saved new best checkpoint.")

    torch.save(model.state_dict(), "checkpoints/picogpt_scratch.pt")
    print("Scratch training complete.")


if __name__ == "__main__":
    main()
