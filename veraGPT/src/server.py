"""
server.py
─────────────────────────────────────────────────────────────────────
Persistent model server for fast multi-prompt inference.
Loads the model once and keeps it in memory for subsequent prompts.

Usage:
    # Start interactive mode (recommended for Colab)
    python server.py

    # Or import and use programmatically:
    from server import ModelServer
    server = ModelServer()
    server.load()  # Load model once
    response = server.ask("What is Newton's second law?")
─────────────────────────────────────────────────────────────────────
"""

import time
from typing import Optional

from config import Config
from device_manager import DeviceManager
from inference_engine import InferenceEngine
from logger import get_logger, setup_logging
from model_loader import ModelLoader

logger = get_logger(__name__)

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class ModelServer:
    """
    Persistent model server that loads once and serves multiple prompts.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config.from_env()
        setup_logging(self.cfg.logging)
        
        self.device_mgr: Optional[DeviceManager] = None
        self.engine: Optional[InferenceEngine] = None
        self._loaded = False

    def load(self) -> None:
        """Load the model into memory (call once at startup)."""
        if self._loaded:
            logger.warning("Model already loaded.")
            return

        logger.info("Loading model (this will take ~60s on first load)...")
        t0 = time.time()

        self.device_mgr = DeviceManager(self.cfg.device)
        loader = ModelLoader(self.cfg, self.device_mgr)
        model, tokeniser = loader.load()

        self.engine = InferenceEngine(
            model=model,
            tokeniser=tokeniser,
            cfg=self.cfg,
            device_mgr=self.device_mgr,
        )

        elapsed = time.time() - t0
        self._loaded = True
        logger.info(f"Model loaded in {elapsed:.2f}s. Ready for prompts!")

    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        show_timing: bool = False,
    ) -> str:
        """
        Generate a response to a single prompt.

        Parameters
        ----------
        prompt : str
            User's question or instruction.
        system_prompt : str, optional
            Custom system prompt (overrides default).
        show_timing : bool
            If True, prints timing stats to console.

        Returns
        -------
        str
            The model's response.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Update system prompt if provided
        if system_prompt:
            self.engine.formatter.system_prompt = system_prompt

        messages = [{"role": "user", "content": prompt}]
        
        if show_timing:
            response, stats = self.engine.generate(messages, return_stats=True)
            print(
                f"\n{YELLOW}[Timing] {stats['elapsed_s']:.2f}s, "
                f"{stats['tokens_per_s']:.2f} tok/s, "
                f"{stats['new_tokens']} tokens{RESET}\n"
            )
            return response
        else:
            return self.engine.generate(messages)

    def interactive(self) -> None:
        """
        Run an interactive REPL loop (for terminal use).
        Type 'quit' or 'exit' to stop.
        """
        if not self._loaded:
            self.load()

        print(f"\n{GREEN}{'='*60}")
        print("  Model Server - Interactive Mode")
        print("  Type your prompt and press Enter.")
        print("  Commands: 'quit', 'exit', 'timing on/off'")
        print(f"{'='*60}{RESET}\n")

        show_timing = False

        while True:
            try:
                user_input = input(f"{CYAN}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("quit", "exit"):
                print("Goodbye!")
                break
            if cmd == "timing on":
                show_timing = True
                print(f"{YELLOW}Timing display enabled.{RESET}")
                continue
            if cmd == "timing off":
                show_timing = False
                print(f"{YELLOW}Timing display disabled.{RESET}")
                continue

            print(f"\n{GREEN}Assistant:{RESET} ", end="", flush=True)
            response = self.ask(user_input, show_timing=show_timing)
            print(response)


def main():
    """Entry point for standalone server usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Persistent model server for fast inference"
    )
    parser.add_argument(
        "--quant", type=int, choices=[4, 8], default=None,
        help="Enable quantization (4-bit or 8-bit).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Target device.",
    )
    args = parser.parse_args()

    # Build config
    cfg = Config.from_env()
    cfg.device.device = args.device
    if args.quant:
        cfg.quantization.enabled = True
        cfg.quantization.load_in_bits = args.quant
    cfg.generation.max_new_tokens = args.max_tokens

    # Start server
    server = ModelServer(cfg)
    server.load()
    server.interactive()


if __name__ == "__main__":
    main()
