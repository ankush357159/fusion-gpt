"""
model_loader.py
─────────────────────────────────────────────────────────────────────
Responsible for downloading / caching, instantiating, and optionally
quantizing the Mistral model and its tokeniser.

    from model_loader import ModelLoader
    loader = ModelLoader(config, device_manager)
    model, tokeniser = loader.load()
─────────────────────────────────────────────────────────────────────
"""

import time
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import Config
from device_manager import DeviceManager
from logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles every step between a model identifier and a live
    (model, tokeniser) pair on the correct device.
    """

    def __init__(self, cfg: Config, device_mgr: DeviceManager):
        self.cfg = cfg
        self.device_mgr = device_mgr
        self._model: torch.nn.Module | None = None
        self._tokeniser: PreTrainedTokenizerBase | None = None

    # ── public API ───────────────────────────────────────────────
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Full load pipeline:
        1. Resolve model path (local or Hub).
        2. Load tokeniser.
        3. Build quantization kwargs if enabled.
        4. Load model.
        5. Optionally attach LoRA adapters.
        Returns (model, tokeniser).
        """
        model_path = self._resolve_path()
        logger.info("Loading model from: %s", model_path)

        t0 = time.time()
        self._tokeniser = self._load_tokeniser(model_path)
        logger.info("Tokeniser loaded in %.2f s", time.time() - t0)

        t0 = time.time()
        self._model = self._load_model(model_path)
        logger.info("Model loaded in %.2f s", time.time() - t0)

        if self.cfg.lora.enabled:
            self._attach_lora()

        return self._model, self._tokeniser

    @property
    def model(self) -> PreTrainedModel | None:
        return self._model

    @property
    def tokeniser(self) -> PreTrainedTokenizerBase | None:
        return self._tokeniser

    # ── private helpers ──────────────────────────────────────────
    def _resolve_path(self) -> str:
        """Return local path if set, else the HF Hub identifier."""
        local = self.cfg.model.local_model_path
        if local:
            import os
            if not os.path.isdir(local):
                raise FileNotFoundError(
                    f"local_model_path '{local}' is not a valid directory."
                )
            logger.info("Using local model path: %s", local)
            return local
        return self.cfg.model.model_name_or_path

    def _load_tokeniser(self, path: str) -> PreTrainedTokenizerBase:
        """Download / cache and return the tokeniser."""
        tokeniser = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            trust_remote_code=False,
        )
        # Mistral models sometimes lack a pad token; reuse EOS
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token
            logger.debug("pad_token was None – set to eos_token (%s)", tokeniser.eos_token)
        return tokeniser

    def _load_model(self, path: str) -> PreTrainedModel:
        """
        Load the causal-LM, optionally in 4-bit or 8-bit via bitsandbytes.
        """
        load_kwargs = self._base_load_kwargs()

        # ── quantization ─────────────────────────────────────────
        if self.cfg.quantization.enabled:
            if not self.device_mgr.is_cuda:
                raise RuntimeError(
                    "Quantization via bitsandbytes requires CUDA. "
                    "Set --device cuda or disable quantization."
                )
            load_kwargs.update(self._quantization_kwargs())
            logger.info(
                "Quantization enabled: %d-bit",
                self.cfg.quantization.load_in_bits,
            )

        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        model.eval()
        return model

    def _base_load_kwargs(self) -> dict:
        """Common kwargs for AutoModelForCausalLM.from_pretrained."""
        device = self.device_mgr.device
        dtype = self.device_mgr.get_torch_dtype(self.cfg.model.torch_dtype)

        kwargs: dict = {
            "torch_dtype": dtype,
            "trust_remote_code": False,
        }

        # When quantizing, HF handles device placement internally
        if self.cfg.quantization.enabled:
            kwargs["device_map"] = "auto"
        else:
            # For non-quantized, place on the resolved device
            # "auto" shards across available GPUs automatically
            if self.device_mgr.is_cpu:
                kwargs["device_map"] = None  # stays on CPU
            else:
                kwargs["device_map"] = "auto"

        return kwargs

    def _quantization_kwargs(self) -> dict:
        """Build the bitsandbytes BitsAndBytesConfig kwargs dict."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for quantization. "
                "Install it: pip install bitsandbytes"
            ) from exc

        bits = self.cfg.quantization.load_in_bits

        if bits == 8:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        elif bits == 4:
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            compute_dtype = compute_dtype_map.get(
                self.cfg.quantization.bnb_4bit_compute_dtype, torch.float16
            )
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.cfg.quantization.bnb_4bit_quant_type,
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {bits}. Use 4 or 8."
            )

        return {"quantization_config": bnb_cfg}

    def _attach_lora(self) -> None:
        """Attach a LoRA adapter to the already-loaded model (fine-tuning path)."""
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType, PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA. Install: pip install peft"
            ) from exc

        # If an adapter path is provided, load it for inference.
        if self.cfg.lora.adapter_path:
            self._model = PeftModel.from_pretrained(
                self._model,
                self.cfg.lora.adapter_path,
            )
            self._model.eval()
            logger.info("LoRA adapter loaded from: %s", self.cfg.lora.adapter_path)
            return

        lora_cfg = PeftLoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

        self._model = get_peft_model(self._model, lora_cfg)
        self._model.print_trainable_parameters()
        logger.info("LoRA adapter attached (rank=%d)", self.cfg.lora.rank)
