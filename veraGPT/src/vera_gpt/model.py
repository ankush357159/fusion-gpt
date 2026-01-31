from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings


class MistralCore:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=settings.hf_token,
            use_fast=True,
        )
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=settings.hf_token,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

    def _build_prompt(self, messages: Iterable[dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback prompt format
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"[{role.upper()}] {content}")
        lines.append("[ASSISTANT]")
        return "\n".join(lines)

    @torch.inference_mode()
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or settings.max_new_tokens,
            temperature=temperature if temperature is not None else settings.temperature,
            top_p=top_p if top_p is not None else settings.top_p,
            do_sample=True,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
