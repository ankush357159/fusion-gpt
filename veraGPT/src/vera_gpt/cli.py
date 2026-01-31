from __future__ import annotations

import argparse

from .model import MistralCore


def main() -> None:
    parser = argparse.ArgumentParser(description="VeraGPT CLI")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    model = MistralCore()
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    output = model.generate(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output)


if __name__ == "__main__":
    main()
