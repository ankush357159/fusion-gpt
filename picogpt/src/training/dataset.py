import numpy as np
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_and_split_datasets(
    tokens_path: str,
    block_size: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
):
    """
    Load tokens and split into train/val/test datasets
    """
    tokens = torch.from_numpy(np.load(tokens_path)).long()
    n = len(tokens)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]

    train_dataset = GPTDataset(train_tokens, block_size)
    val_dataset = GPTDataset(val_tokens, block_size)
    test_dataset = GPTDataset(test_tokens, block_size)

    print(f"Total tokens: {n:,}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")
    print(f"Test tokens:  {len(test_tokens):,}")

    return train_dataset, val_dataset, test_dataset
