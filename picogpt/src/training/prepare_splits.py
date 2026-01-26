import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch

from src.tokenizer.tokenizer import TiktokenTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test token splits for PicoGPT."
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=str(RAW_DIR / "combined_novels.txt"),
        help="Path to raw text file to tokenize when input_tokens is not provided.",
    )
    parser.add_argument(
        "--input_tokens",
        type=str,
        default=None,
        help="Optional path to pre-tokenized file (.npy or .pt).",
    )
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tokens before splitting (not recommended for LM).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="Output directory for split .npy files.",
    )
    return parser.parse_args()


def _load_tokens_from_npy(path: Path) -> np.ndarray:
    tokens = np.load(path)
    if tokens.ndim != 1:
        tokens = tokens.reshape(-1)
    return tokens.astype(np.int64)


def _load_tokens_from_pt(path: Path) -> np.ndarray:
    tokens = torch.load(path, map_location="cpu")
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.flatten().cpu().numpy()
    else:
        tokens = np.asarray(tokens).reshape(-1)
    return tokens.astype(np.int64)


def _tokenize_text(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    tokenizer = TiktokenTokenizer()
    tokens = tokenizer.encode(text)
    return np.asarray(tokens, dtype=np.int64)


def _validate_splits(train_split: float, val_split: float, test_split: float) -> None:
    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Splits must sum to 1.0. Got train={train_split}, val={val_split}, test={test_split}"
        )
    if not (0 < train_split < 1 and 0 <= val_split < 1 and 0 <= test_split < 1):
        raise ValueError("Split values must be in [0, 1).")


def _split_tokens(
    tokens: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
    shuffle: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shuffle:
        rng = np.random.default_rng(seed)
        tokens = tokens.copy()
        rng.shuffle(tokens)

    n = len(tokens)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]

    return train_tokens, val_tokens, test_tokens


def _save_tokens(output_dir: Path, name: str, tokens: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_tokens.npy"
    np.save(output_path, tokens)
    return output_path


def main():
    args = parse_args()
    _validate_splits(args.train_split, args.val_split, args.test_split)

    input_tokens_path: Optional[Path] = (
        Path(args.input_tokens) if args.input_tokens else None
    )

    if input_tokens_path is not None:
        if not input_tokens_path.exists():
            raise FileNotFoundError(f"Input tokens file not found: {input_tokens_path}")
        if input_tokens_path.suffix == ".npy":
            tokens = _load_tokens_from_npy(input_tokens_path)
        elif input_tokens_path.suffix == ".pt":
            tokens = _load_tokens_from_pt(input_tokens_path)
        else:
            raise ValueError("input_tokens must be a .npy or .pt file")
    else:
        input_text_path = Path(args.input_text)
        if not input_text_path.exists():
            raise FileNotFoundError(f"Input text file not found: {input_text_path}")
        tokens = _tokenize_text(input_text_path)

    train_tokens, val_tokens, test_tokens = _split_tokens(
        tokens,
        args.train_split,
        args.val_split,
        args.test_split,
        args.shuffle,
        args.seed,
    )

    output_dir = Path(args.output_dir)
    train_path = _save_tokens(output_dir, "train", train_tokens)
    val_path = _save_tokens(output_dir, "val", val_tokens)
    test_path = _save_tokens(output_dir, "test", test_tokens)

    print("Split complete:")
    print(f"- Train tokens: {train_path} ({len(train_tokens):,})")
    print(f"- Val tokens:   {val_path} ({len(val_tokens):,})")
    print(f"- Test tokens:  {test_path} ({len(test_tokens):,})")


if __name__ == "__main__":
    main()
