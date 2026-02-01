"""
inference_engine.py
─────────────────────────────────────────────────────────────────────
The single entry point for running text generation.  Supports both
blocking (full response at once) and streaming (token-by-token via
a generator).

    from inference_engine import InferenceEngine
    engine = InferenceEngine(model, tokeniser, config, device_manager)
    response = engine.generate(messages)            # blocking
    for token in engine.generate_stream(messages):  # streaming
        print(token, end="", flush=True)
─────────────────────────────────────────────────────────────────────
"""

import time
from typing import Generator, List

import torch

from config import Config
from device_manager import DeviceManager
from logger import get_logger
from prompt_formatter import PromptFormatter

logger = get_logger(__name__)


class InferenceEngine:
    """Wraps model + tokeniser and exposes clean generate / stream APIs."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokeniser,
        cfg: Config,
        device_mgr: DeviceManager,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.tokeniser = tokeniser
        self.cfg = cfg
        self.device_mgr = device_mgr
        self.formatter = PromptFormatter(
            system_prompt=system_prompt or ""
        )

    # ── blocking generation ──────────────────────────────────────
    @torch.inference_mode()
    def generate(self, messages: List[dict], return_stats: bool = False):
        """
        Full-response generation.

        Parameters
        ----------
        messages : list[dict]
            Chat history as [{"role": ..., "content": ...}, …].
        return_stats : bool
            If True, returns (response, stats_dict) instead of just response.

        Returns
        -------
        str or tuple
            The model's reply (assistant turn only, no prompt echo).
            If return_stats=True, returns (response, {"new_tokens", "elapsed_s", "tokens_per_s"}).
        """
        prompt = self.formatter.format(messages)
        input_ids, attention_mask = self._tokenize(prompt)

        gen_cfg = self._build_generation_config()

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_cfg,
            )
        elapsed = time.time() - t0

        # Slice off the echoed prompt tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokeniser.decode(new_tokens, skip_special_tokens=True)

        logger.info(
            "Generation complete: %d new tokens in %.2f s (%.1f tok/s)",
            len(new_tokens),
            elapsed,
            len(new_tokens) / elapsed if elapsed > 0 else 0,
        )
        
        if return_stats:
            stats = {
                "new_tokens": len(new_tokens),
                "elapsed_s": elapsed,
                "tokens_per_s": (len(new_tokens) / elapsed) if elapsed > 0 else 0,
            }
            return response.strip(), stats

        return response.strip()

    # ── streaming generation ─────────────────────────────────────
    @torch.no_grad()
    def generate_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        """
        Token-by-token streaming generator.

        Yields
        ------
        str
            Each decoded token as it is produced.
        """
        prompt = self.formatter.format(messages)
        input_ids = self._tokenize(prompt)

        # Cache for incremental decoding
        generated_ids = input_ids.clone()

        max_new = self.cfg.generation.max_new_tokens
        greedy = self.cfg.generation.greedy or not self.cfg.generation.do_sample

        for _ in range(max_new):
            # Single forward pass for the next token
            with torch.no_grad():
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            rep_pen = self.cfg.generation.repetition_penalty
            if rep_pen != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, rep_pen
                )

            # Sampling or greedy
            if greedy:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = self._sample_token(next_token_logits)

            # Append new token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Decode and yield
            token_str = self.tokeniser.decode(next_token[0], skip_special_tokens=True)
            if token_str:
                yield token_str

            # Stop on EOS
            if next_token.item() == self.tokeniser.eos_token_id:
                logger.debug("EOS token encountered – stream ended.")
                break

    # ── private helpers ──────────────────────────────────────────
    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text to input_ids + attention_mask on the correct device."""
        inputs = self.tokeniser(
            text,
            return_tensors="pt",
            add_special_tokens=False,  # We already include <s> in the template
        )
        return (
            inputs["input_ids"].to(self.device_mgr.device),
            inputs["attention_mask"].to(self.device_mgr.device),
        )

    def _build_generation_config(self) -> dict:
        """Map our GenerationConfig dataclass to HF generate() kwargs."""
        gen = self.cfg.generation
        greedy = gen.greedy or not gen.do_sample

        kwargs: dict = {
            "max_new_tokens": gen.max_new_tokens,
            "do_sample": not greedy,
            "pad_token_id": self.tokeniser.pad_token_id,
            "eos_token_id": self.tokeniser.eos_token_id,
            "repetition_penalty": gen.repetition_penalty,
            "use_cache": True,
        }

        if not greedy:
            kwargs["temperature"] = gen.temperature
            kwargs["top_p"] = gen.top_p
            kwargs["top_k"] = gen.top_k

        return kwargs

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature + top-k + top-p then sample one token."""
        gen = self.cfg.generation

        # Temperature
        logits = logits / max(gen.temperature, 1e-8)

        # Top-k
        if gen.top_k > 0:
            top_k_values, _ = torch.topk(logits, gen.top_k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits[logits < threshold] = float("-inf")

        # Top-p (nucleus)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Remove tokens above the cumulative threshold
        sorted_indices_to_remove = cumulative_probs - torch.nn.functional.softmax(
            sorted_logits, dim=-1
        ) > gen.top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        # Scatter back
        logits.scatter_(-1, sorted_indices, sorted_logits)

        # Sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        prev_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Penalise logits of tokens that already appeared in the sequence."""
        unique_ids = prev_ids[0].unique()
        for token_id in unique_ids:
            if logits[0, token_id] < 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits
