"""
chat_history.py
─────────────────────────────────────────────────────────────────────
Manages in-memory conversation state AND optional JSON persistence
so that chats survive restarts.

    from chat_history import ChatHistory
    history = ChatHistory(session_id="my_session", persist_dir="chat_history")
    history.add_user("Hello")
    history.add_assistant("Hi there!")
    msgs = history.get_messages()     # list[dict]
    history.save()                    # writes to disk
─────────────────────────────────────────────────────────────────────
"""

import json
import os
import time
import uuid
from typing import List, Optional

from logger import get_logger

logger = get_logger(__name__)


class ChatHistory:
    """
    Thread-safe-ish (single-threaded assumption) container for one
    conversation's message list.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.persist_dir = persist_dir
        self._messages: List[dict] = []

        # Try to resume from disk if a file already exists
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_if_exists()

    # ── public ───────────────────────────────────────────────────
    def add_user(self, content: str) -> None:
        """Append a user message."""
        self._messages.append(self._make_msg("user", content))
        logger.debug("[%s] User message added (%d chars)", self.session_id, len(content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self._messages.append(self._make_msg("assistant", content))
        logger.debug("[%s] Assistant message added (%d chars)", self.session_id, len(content))

    def get_messages(self) -> List[dict]:
        """Return a shallow copy of the full message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Wipe the in-memory history (and the file, if persisting)."""
        self._messages.clear()
        logger.info("[%s] Chat history cleared.", self.session_id)
        if self.persist_dir:
            path = self._file_path()
            if os.path.exists(path):
                os.remove(path)

    def save(self) -> Optional[str]:
        """
        Persist current history to a JSON file.
        Returns the file path, or None if persist_dir is not set.
        """
        if not self.persist_dir:
            return None

        path = self._file_path()
        payload = {
            "session_id": self.session_id,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "messages": self._messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[%s] History saved to %s (%d messages)", self.session_id, path, len(self._messages))
        return path

    @property
    def length(self) -> int:
        """Number of messages in this session."""
        return len(self._messages)

    @property
    def last_user_message(self) -> Optional[str]:
        """Content of the most recent user message, or None."""
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    # ── private ──────────────────────────────────────────────────
    def _file_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.session_id}.json")

    def _load_if_exists(self) -> None:
        path = self._file_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._messages = data.get("messages", [])
                logger.info(
                    "[%s] Resumed session from %s (%d messages)",
                    self.session_id,
                    path,
                    len(self._messages),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("[%s] Could not load history: %s", self.session_id, exc)
                self._messages = []

    @staticmethod
    def _make_msg(role: str, content: str) -> dict:
        return {
            "role": role,
            "content": content.strip(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
