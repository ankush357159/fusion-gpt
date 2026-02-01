"""
prompt_formatter.py
─────────────────────────────────────────────────────────────────────
Converts a list of chat messages into a single string that Mistral's
tokeniser / model expects.

Mistral-Instruct uses a simple BOS-delimited format:
    <s>[INST] {user_turn} [/INST] {assistant_turn}</s>[INST] …

This module also lets you plug in a *system prompt* that is prepended
to the very first user turn (Mistral v0.2 does not have a native
<|system|> token, so this is the canonical approach).

    from prompt_formatter import PromptFormatter
    fmt = PromptFormatter(system_prompt="You are a helpful assistant.")
    text = fmt.format(messages)   # list of {"role": ..., "content": ...}
─────────────────────────────────────────────────────────────────────
"""

from typing import List

from logger import get_logger

logger = get_logger(__name__)

# Token constants for Mistral-Instruct
BOS   = "<s>"
EOS   = "</s>"
B_INST = "[INST]"
E_INST = "[/INST]"

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as helpfully as possible. "
    "If a question does not make any sense or is not factually coherent, "
    "explain why instead of answering something incorrect."
)


class PromptFormatter:
    """
    Stateless formatter: turns a list of {role, content} dicts into the
    exact string layout Mistral-Instruct expects for multi-turn chat.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt.strip()

    # ── public ───────────────────────────────────────────────────
    def format(self, messages: List[dict]) -> str:
        """
        Parameters
        ----------
        messages : list of dicts
            Each dict must have "role" (user | assistant) and "content".
            An optional leading dict with role="system" is ignored here
            because we inject self.system_prompt into the first user turn.

        Returns
        -------
        str
            The fully formatted prompt string ready for tokenisation.
        """
        # Filter out any explicit system message – we use self.system_prompt
        filtered = [m for m in messages if m["role"] != "system"]

        if not filtered:
            raise ValueError("At least one user message is required.")

        # Validate alternation starting with user
        self._validate_turn_order(filtered)

        return self._build_prompt(filtered)

    def format_single(self, user_input: str) -> str:
        """Convenience: format a single-turn user prompt."""
        return self.format([{"role": "user", "content": user_input}])

    # ── private ──────────────────────────────────────────────────
    def _build_prompt(self, messages: List[dict]) -> str:
        """
        Walk through message pairs and emit the Mistral chat template.

        Structure per turn pair:
            <s>[INST] {user} [/INST] {assistant}</s>
        The last turn (always a user turn awaiting a response) omits
        the closing </s> and assistant text.
        """
        prompt_parts: List[str] = []
        # Group into (user, assistant?) pairs
        pairs = self._pair_messages(messages)

        for idx, (user_msg, assistant_msg) in enumerate(pairs):
            user_content = user_msg["content"].strip()

            # Prepend system prompt to the very first user turn only
            if idx == 0 and self.system_prompt:
                user_content = f"{self.system_prompt}\n\n{user_content}"

            if assistant_msg:
                # Complete turn (used for conversation history)
                turn = (
                    f"{BOS}{B_INST} {user_content} {E_INST} "
                    f"{assistant_msg['content'].strip()}{EOS}"
                )
            else:
                # Final / open turn – model generates from here
                turn = f"{BOS}{B_INST} {user_content} {E_INST}"

            prompt_parts.append(turn)

        prompt = "".join(prompt_parts)
        logger.debug("Formatted prompt (%d chars):\n%s", len(prompt), prompt)
        return prompt

    @staticmethod
    def _pair_messages(
        messages: List[dict],
    ) -> List[tuple]:
        """
        Pair consecutive (user, assistant) messages.
        The last element may have assistant=None if the conversation
        ends on a user turn (the normal case for inference).
        """
        pairs = []
        i = 0
        while i < len(messages):
            user = messages[i]
            assistant = messages[i + 1] if (i + 1 < len(messages)) else None
            pairs.append((user, assistant))
            i += 2  # step over both user and assistant
        return pairs

    @staticmethod
    def _validate_turn_order(messages: List[dict]) -> None:
        """Ensure messages strictly alternate user → assistant → user …"""
        expected_roles = ["user", "assistant"]
        for i, msg in enumerate(messages):
            expected = expected_roles[i % 2]
            if msg["role"] != expected:
                raise ValueError(
                    f"Message at index {i} has role='{msg['role']}' "
                    f"but expected '{expected}'. Messages must alternate "
                    "user / assistant, starting with user."
                )
