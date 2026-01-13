import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np

# This file needs correction
class C4Dataset(Dataset):
    """
    Custom Dataset class for loading C4 dataset from TensorFlow Datasets
    for GPT-2 training.
    """

    def __init__(self, max_length=512, max_size_mb=1, split="train"):
        """
        Initialize the C4 Dataset.

        Args:
            max_length: Maximum sequence length for tokenization
            max_size_mb: Maximum dataset size in MB to load
            split: Dataset split ('train', 'validation')
        """
        self.max_length = max_length
        self.max_size_mb = max_size_mb
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load C4 dataset
        print(f"Loading C4 dataset (up to {max_size_mb}MB)...")
        self.texts = self._load_c4_data(split)
        print(f"Loaded {len(self.texts)} documents")

    def _load_c4_data(self, split):
        """Load and filter C4 data up to specified size."""
        texts = []
        total_size = 0
        max_size_bytes = self.max_size_mb * 1024 * 1024

        # Load C4 dataset from TensorFlow Datasets
        ds = tfds.load("c4/en", split=split, streaming=True)

        for example in ds:
            text = example["text"].numpy().decode("utf-8")
            text_size = len(text.encode("utf-8"))

            if total_size + text_size > max_size_bytes:
                break

            texts.append(text)
            total_size += text_size

            if len(texts) % 100 == 0:
                print(f"Loaded {len(texts)} docs, {total_size / (1024*1024):.2f}MB")

        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Returns:
            dict: Dictionary containing input_ids and attention_mask
        """
        text = self.texts[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by return_tensors='pt'
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For language modeling, labels = input_ids
        }


def create_dataloader(
    batch_size=4,
    max_length=512,
    max_size_mb=1,
    split="train",
    num_workers=0,
    shuffle=True,
):
    """
    Create a DataLoader for the C4 dataset.

    Args:
        batch_size: Batch size for training
        max_length: Maximum sequence length
        max_size_mb: Maximum dataset size in MB
        split: Dataset split ('train', 'validation')
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = C4Dataset(max_length=max_length, max_size_mb=max_size_mb, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    # Create dataloader
    train_loader = create_dataloader(
        batch_size=4,
        max_length=512,
        max_size_mb=1,
        split="train[:1%]",  # Use only 1% of training data
        shuffle=True,
    )

    # Test the dataloader
    print("\nTesting DataLoader:")
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")

        # Decode first example in batch
        decoded = train_loader.dataset.tokenizer.decode(
            batch["input_ids"][0], skip_special_tokens=True
        )
        print(f"  First example (truncated): {decoded[:100]}...")

        if i >= 2:  # Show only 3 batches
            break

    print(f"\nTotal batches: {len(train_loader)}")
