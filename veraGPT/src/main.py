"""
main.py
─────────────────────────────────────────────────────────────────────
Entry point.  Parses CLI flags, builds a Config, and launches the
chatbot.

Usage examples:
    # Basic - uses all defaults (auto device, no quantization)
    python main.py

    # Enable 4-bit quantization for lower VRAM usage
    python main.py --quant 4

    # Custom system prompt + greedy decoding
    python main.py --system "You are a pirate." --greedy

    # Load from a local directory instead of HuggingFace Hub
    python main.py --local-path /data/mistral-7b-instruct

    # Non-interactive: single prompt and exit
    python main.py --prompt "What is the capital of France?"
─────────────────────────────────────────────────────────────────────
"""

import argparse
import sys

from chatbot import Chatbot
from config import Config
from inference_engine import InferenceEngine
from logger import setup_logging
from model_loader import ModelLoader
from device_manager import DeviceManager


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive chatbot using Mistral-7B-Instruct-v0.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model ────────────────────────────────────────────────────
    p.add_argument(
        "--model", default=None,
        help="HuggingFace model id or local path (overrides config default).",
    )
    p.add_argument(
        "--local-path", default=None,
        help="Path to a locally-saved model directory.",
    )

    # ── Device ───────────────────────────────────────────────────
    p.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Target device for inference (default: auto).",
    )

    # ── Quantization ─────────────────────────────────────────────
    p.add_argument(
        "--quant", type=int, choices=[4, 8], default=None,
        help="Enable 4-bit or 8-bit quantization via bitsandbytes.",
    )

    # ── LoRA ─────────────────────────────────────────────────────
    p.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to a PEFT LoRA adapter to load for inference.",
    )
    p.add_argument(
        "--lora-rank", type=int, default=None,
        help="LoRA rank (for attaching a fresh adapter).",
    )
    p.add_argument(
        "--lora-alpha", type=int, default=None,
        help="LoRA alpha (for attaching a fresh adapter).",
    )
    p.add_argument(
        "--lora-dropout", type=float, default=None,
        help="LoRA dropout (for attaching a fresh adapter).",
    )
    p.add_argument(
        "--lora-target-modules", type=str, default=None,
        help="Comma-separated target modules for LoRA (e.g. q_proj,v_proj).",
    )
    p.add_argument(
        "--lora-enable", action="store_true",
        help="Enable LoRA with default settings (for fine-tuning workflows).",
    )

    # ── Generation ───────────────────────────────────────────────
    p.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum number of new tokens to generate (default: 512).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7). Lower = more deterministic.",
    )
    p.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus (top-p) sampling threshold (default: 0.9).",
    )
    p.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling pool size (default: 50).",
    )
    p.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding (overrides temperature / top-p / top-k).",
    )

    # ── Chat behaviour ───────────────────────────────────────────
    p.add_argument(
        "--system", type=str, default=None,
        help="System prompt to prepend to the first user message.",
    )
    p.add_argument(
        "--session-id", type=str, default=None,
        help="Resume or start a named session (default: new UUID).",
    )
    p.add_argument(
        "--no-stream", action="store_true",
        help="Disable token streaming (return full response at once).",
    )

    # ── One-shot mode ────────────────────────────────────────────
    p.add_argument(
        "--prompt", type=str, default=None,
        help="If provided, run a single prompt and exit (non-interactive).",
    )
    p.add_argument(
        "--timing", action="store_true",
        help="Print response timing metrics (tokens/sec, latency).",
    )

    # ── Logging ──────────────────────────────────────────────────
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return p


def args_to_config(args: argparse.Namespace) -> Config:
    """Merge CLI arguments into a Config dataclass."""
    cfg = Config.from_env()  # start with env-var overrides

    # Model
    if args.model:
        cfg.model.model_name_or_path = args.model
    if args.local_path:
        cfg.model.local_model_path = args.local_path

    # Device
    cfg.device.device = args.device

    # Quantization
    if args.quant:
        cfg.quantization.enabled = True
        cfg.quantization.load_in_bits = args.quant

    # Generation
    cfg.generation.max_new_tokens = args.max_tokens
    cfg.generation.temperature = args.temperature
    cfg.generation.top_p = args.top_p
    cfg.generation.top_k = args.top_k
    cfg.generation.greedy = args.greedy

    # LoRA
    if args.lora_enable:
        cfg.lora.enabled = True
    if args.lora_path:
        cfg.lora.enabled = True
        cfg.lora.adapter_path = args.lora_path
    if args.lora_rank is not None:
        cfg.lora.enabled = True
        cfg.lora.rank = args.lora_rank
    if args.lora_alpha is not None:
        cfg.lora.enabled = True
        cfg.lora.alpha = args.lora_alpha
    if args.lora_dropout is not None:
        cfg.lora.enabled = True
        cfg.lora.dropout = args.lora_dropout
    if args.lora_target_modules:
        cfg.lora.enabled = True
        cfg.lora.target_modules = [
            t.strip() for t in args.lora_target_modules.split(",") if t.strip()
        ]

    # Logging
    cfg.logging.log_level = args.log_level

    return cfg


def run_single_prompt(cfg: Config, prompt: str, system_prompt: str | None, show_timing: bool = False) -> None:
    """Non-interactive: load model, generate one response, print, exit."""
    setup_logging(cfg.logging)

    device_mgr = DeviceManager(cfg.device)
    loader = ModelLoader(cfg, device_mgr)
    model, tokeniser = loader.load()

    engine = InferenceEngine(
        model=model,
        tokeniser=tokeniser,
        cfg=cfg,
        device_mgr=device_mgr,
        system_prompt=system_prompt,
    )

    messages = [{"role": "user", "content": prompt}]
    
    if show_timing:
        response, stats = engine.generate(messages, return_stats=True)
        print(response)
        print(
            f"\n{'='*60}\n"
            f"TIMING STATS\n"
            f"{'='*60}\n"
            f"New tokens: {stats['new_tokens']}\n"
            f"Elapsed: {stats['elapsed_s']:.2f} s\n"
            f"Speed: {stats['tokens_per_s']:.2f} tok/s\n"
            f"{'='*60}"
        )
    else:
        response = engine.generate(messages)
        print(response)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    # ── Non-interactive single-prompt mode ───────────────────────
    if args.prompt:
        run_single_prompt(cfg, args.prompt, args.system, show_timing=args.timing)
        return

    # ── Interactive chatbot loop ─────────────────────────────────
    bot = Chatbot(
        cfg=cfg,
        system_prompt=args.system,
        session_id=args.session_id,
        stream=not args.no_stream,
    )
    bot.chat()


if __name__ == "__main__":
    main()
