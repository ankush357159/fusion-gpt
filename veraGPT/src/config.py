"""
config.py
─────────────────────────────────────────────────────────────────────
Central configuration for the entire LLM pipeline.
All tunables live here so every other module just does:
    from config import Config
─────────────────────────────────────────────────────────────────────
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Pretrained model & tokeniser path / hub id."""
    # Default to TinyLlama (fastest, works on CPU, beginner-friendly)
    # Use Config.from_preset() for other models: 'tiny', 'phi2', 'phi3', 'mistral', 'llama13'
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Set to a local directory to load from disk instead of HF Hub
    local_model_path: Optional[str] = None
    torch_dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    # Model preset name (if using presets: 'tiny', 'phi2', 'phi3', 'mistral', 'llama13')
    preset: Optional[str] = "tiny"  # Default preset


@dataclass
class GenerationConfig:
    """Decoding / sampling hyper-parameters."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    # If True, apply greedy decoding (overrides do_sample)
    greedy: bool = False


@dataclass
class QuantizationConfig:
    """Optional quantization to reduce GPU memory footprint."""
    enabled: bool = False
    # "4bit" or "8bit"
    load_in_bits: int = 4
    # Only relevant for 4-bit
    bnb_4bit_compute_dtype: str = "float16"  # "float16" | "bfloat16"
    bnb_4bit_quant_type: str = "nf4"         # "nf4" | "fp4"


@dataclass
class LoraConfig:
    """Optional LoRA fine-tuning knobs (used only when fine-tuning)."""
    enabled: bool = False
    # Optional: path to a PEFT LoRA adapter for inference
    adapter_path: Optional[str] = None
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class DeviceConfig:
    """Hardware selection."""
    # "auto" picks the best available (CUDA > MPS > CPU)
    device: str = "auto"
    # Number of GPUs for model-parallel (0 = all available)
    num_gpus: int = 0


@dataclass
class LoggingConfig:
    """Logging verbosity and output paths."""
    log_level: str = "INFO"       # DEBUG | INFO | WARNING | ERROR
    log_file: Optional[str] = "llm_app.log"
    # Directory where chat history JSON files are saved
    history_dir: str = "chat_history"


# ─── Master singleton ────────────────────────────────────────────
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── Convenience factory ──────────────────────────────────────
    @classmethod
    def from_env(cls) -> "Config":
        """
        Override defaults from environment variables.
        Supported env vars (all optional):
            LLM_MODEL_PATH          - HF hub id or local path
            LLM_MODEL_PRESET        - 'tiny', 'phi2', 'phi3', 'mistral', 'llama13'
            LLM_MAX_NEW_TOKENS      - int
            LLM_TEMPERATURE         - float
            LLM_QUANT_ENABLED       - "1" to enable 4/8-bit quantization
            LLM_QUANT_BITS          - 4 or 8
            LLM_LORA_ENABLED        - "1" to enable LoRA
            LLM_LORA_PATH           - path to PEFT LoRA adapter
            LLM_LORA_RANK           - int
            LLM_LORA_ALPHA          - int
            LLM_LORA_DROPOUT        - float
            LLM_LORA_TARGET_MODULES - comma-separated list (e.g. q_proj,v_proj)
            LLM_DEVICE              - "cpu" | "cuda" | "mps" | "auto"
            LLM_LOG_LEVEL           - DEBUG | INFO | WARNING | ERROR
        """
        cfg = cls()

        # Model preset takes precedence
        model_preset = os.getenv("LLM_MODEL_PRESET")
        if model_preset:
            try:
                from model_presets import get_preset
                preset = get_preset(model_preset)
                cfg.model.model_name_or_path = preset.model_id
                cfg.model.preset = model_preset
                if preset.quantization_recommended:
                    cfg.quantization.enabled = True
                    cfg.quantization.load_in_bits = 4
            except (ImportError, ValueError) as e:
                print(f"Warning: Could not load preset '{model_preset}': {e}")

        model_path = os.getenv("LLM_MODEL_PATH")
        if model_path:
            cfg.model.model_name_or_path = model_path

        max_tokens = os.getenv("LLM_MAX_NEW_TOKENS")
        if max_tokens:
            cfg.generation.max_new_tokens = int(max_tokens)

        temperature = os.getenv("LLM_TEMPERATURE")
        if temperature:
            cfg.generation.temperature = float(temperature)

        quant_enabled = os.getenv("LLM_QUANT_ENABLED")
        if quant_enabled == "1":
            cfg.quantization.enabled = True

        quant_bits = os.getenv("LLM_QUANT_BITS")
        if quant_bits:
            cfg.quantization.load_in_bits = int(quant_bits)

        lora_enabled = os.getenv("LLM_LORA_ENABLED")
        if lora_enabled == "1":
            cfg.lora.enabled = True

        lora_path = os.getenv("LLM_LORA_PATH")
        if lora_path:
            cfg.lora.enabled = True
            cfg.lora.adapter_path = lora_path

        lora_rank = os.getenv("LLM_LORA_RANK")
        if lora_rank:
            cfg.lora.enabled = True
            cfg.lora.rank = int(lora_rank)

        lora_alpha = os.getenv("LLM_LORA_ALPHA")
        if lora_alpha:
            cfg.lora.enabled = True
            cfg.lora.alpha = int(lora_alpha)

        lora_dropout = os.getenv("LLM_LORA_DROPOUT")
        if lora_dropout:
            cfg.lora.enabled = True
            cfg.lora.dropout = float(lora_dropout)

        lora_targets = os.getenv("LLM_LORA_TARGET_MODULES")
        if lora_targets:
            cfg.lora.enabled = True
            cfg.lora.target_modules = [
                t.strip() for t in lora_targets.split(",") if t.strip()
            ]

        device = os.getenv("LLM_DEVICE")
        if device:
            cfg.device.device = device

        log_level = os.getenv("LLM_LOG_LEVEL")
        if log_level:
            cfg.logging.log_level = log_level.upper()

        return cfg

    @classmethod
    def from_preset(cls, preset_name: str, enable_quantization: bool = None) -> "Config":
        """
        Create config from a model preset.
        
        Args:
            preset_name: One of 'tiny', 'phi2', 'phi3', 'mistral', 'llama13'
            enable_quantization: Override quantization setting (None = use preset default)
        
        Returns:
            Config object configured for the preset
        """
        from model_presets import get_preset
        
        cfg = cls()
        preset = get_preset(preset_name)
        
        cfg.model.model_name_or_path = preset.model_id
        cfg.model.preset = preset_name
        
        # Apply recommended quantization unless overridden
        if enable_quantization is not None:
            cfg.quantization.enabled = enable_quantization
        else:
            cfg.quantization.enabled = preset.quantization_recommended
        
        if cfg.quantization.enabled:
            cfg.quantization.load_in_bits = 4
        
        return cfg
