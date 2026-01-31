import os
from dataclasses import dataclass


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


@dataclass(frozen=True)
class Settings:
    model_name: str = _get_env("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2") or "mistralai/Mistral-7B-Instruct-v0.2"
    hf_token: str | None = _get_env("HF_TOKEN")
    max_new_tokens: int = int(_get_env("MAX_NEW_TOKENS", "256") or 256)
    temperature: float = float(_get_env("TEMPERATURE", "0.7") or 0.7)
    top_p: float = float(_get_env("TOP_P", "0.95") or 0.95)


settings = Settings()
