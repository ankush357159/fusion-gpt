"""
chatbot.py
─────────────────────────────────────────────────────────────────────
Interactive command-line chatbot.  Wires together every module and
gives the user a REPL-style conversation loop.

Special commands (type at the prompt):
    /quit  or  /exit   - end the session
    /clear             - wipe conversation history
    /save              - manually save chat history to disk
    /history           - print the current conversation so far
    /help              - show this list
    /stream            - toggle streaming mode on / off
────────────────────────────────────────────────────────────────────
"""

import sys
from typing import Optional

from chat_history import ChatHistory
from config import Config
from device_manager import DeviceManager
from inference_engine import InferenceEngine
from logger import get_logger, setup_logging
from model_loader import ModelLoader

logger = get_logger(__name__)

# ── colours (ANSI – ignored on Windows cmd) ─────────────────────
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class Chatbot:
    """
    Orchestrator: owns the config, device manager, model, inference
    engine, and chat history.  Exposes a blocking `chat()` loop.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream: bool = True,
    ):
        self.cfg = cfg or Config.from_env()
        self.stream = stream

        # ── bootstrap ────────────────────────────────────────────
        setup_logging(self.cfg.logging)

        self.device_mgr = DeviceManager(self.cfg.device)
        loader = ModelLoader(self.cfg, self.device_mgr)
        self.model, self.tokeniser = loader.load()

        self.engine = InferenceEngine(
            model=self.model,
            tokeniser=self.tokeniser,
            cfg=self.cfg,
            device_mgr=self.device_mgr,
            system_prompt=system_prompt,
        )

        self.history = ChatHistory(
            session_id=session_id,
            persist_dir=self.cfg.logging.history_dir,
        )

    # ── main loop ────────────────────────────────────────────────
    def chat(self) -> None:
        """Run the interactive REPL."""
        self._print_banner()

        while True:
            try:
                user_input = input(f"\n{CYAN}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            # ── special commands ─────────────────────────────────
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                self.history.save()
                print("Session saved. Goodbye!")
                break
            if cmd == "/clear":
                self.history.clear()
                print(f"{YELLOW}History cleared.{RESET}")
                continue
            if cmd == "/save":
                path = self.history.save()
                print(f"{YELLOW}Saved to: {path}{RESET}")
                continue
            if cmd == "/history":
                self._print_history()
                continue
            if cmd == "/stream":
                self.stream = not self.stream
                state = "ON" if self.stream else "OFF"
                print(f"{YELLOW}Streaming: {state}{RESET}")
                continue
            if cmd == "/help":
                self._print_help()
                continue

            # ── normal turn ──────────────────────────────────────
            self.history.add_user(user_input)
            messages = self.history.get_messages()

            print(f"\n{GREEN}Assistant:{RESET} ", end="", flush=True)

            if self.stream:
                response = self._generate_streaming(messages)
            else:
                response = self.engine.generate(messages)
                print(response)

            self.history.add_assistant(response)
            self.history.save()  # auto-save after every turn

    # ── private ──────────────────────────────────────────────────
    def _generate_streaming(self, messages: list) -> str:
        """Stream tokens to stdout and return the full response."""
        chunks = []
        for token in self.engine.generate_stream(messages):
            print(token, end="", flush=True)
            chunks.append(token)
        print()  # newline after stream ends
        return "".join(chunks)

    # ── pretty-print helpers ─────────────────────────────────────
    @staticmethod
    def _print_banner() -> None:
        print("=" * 60)
        print("  Mistral-7B-Instruct Chatbot")
        print("  Type /help for available commands.")
        print("=" * 60)

    def _print_history(self) -> None:
        print(f"\n{YELLOW}--- Conversation History ---{RESET}")
        for msg in self.history.get_messages():
            role = msg["role"].capitalize()
            color = CYAN if role == "User" else GREEN
            print(f"  {color}{role}:{RESET} {msg['content']}")
        print(f"{YELLOW}--- End ---{RESET}\n")

    @staticmethod
    def _print_help() -> None:
        print(
            f"\n{YELLOW}Available commands:{RESET}\n"
            "  /quit      - exit and save\n"
            "  /exit      - same as /quit\n"
            "  /clear     - wipe conversation history\n"
            "  /save      - manually save history to disk\n"
            "  /history   - display current conversation\n"
            "  /stream    - toggle streaming on/off\n"
            "  /help      - show this help message\n"
        )
