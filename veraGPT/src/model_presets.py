"""
model_presets.py
─────────────────────────────────────────────────────────────────────
Predefined model configurations for different use cases.
Makes it easy to switch between models based on hardware and speed requirements.
─────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelPreset:
    """Configuration preset for a specific model."""
    name: str
    model_id: str
    description: str
    min_ram_gb: float
    min_vram_gb: float
    speed_cpu: str
    speed_gpu: str
    quality: str
    recommended_for: str
    quantization_recommended: bool = False


# ─── Available Model Presets ─────────────────────────────────────
PRESETS: Dict[str, ModelPreset] = {
    "tiny": ModelPreset(
        name="TinyLlama",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Smallest model, fastest responses, basic quality",
        min_ram_gb=2.0,
        min_vram_gb=2.0,
        speed_cpu="2-5 min",
        speed_gpu="5-10 sec (15-20 tok/s)",
        quality="Basic",
        recommended_for="CPU runtime, quick tests, fast responses needed",
        quantization_recommended=True,
    ),
    
    "phi2": ModelPreset(
        name="Phi-2",
        model_id="microsoft/phi-2",
        description="Small but capable, good balance",
        min_ram_gb=6.0,
        min_vram_gb=6.0,
        speed_cpu="Not recommended (OOM risk)",
        speed_gpu="8-15 sec (8-12 tok/s)",
        quality="Good",
        recommended_for="T4 GPU, balanced speed/quality",
        quantization_recommended=True,
    ),
    
    "phi3": ModelPreset(
        name="Phi-3-mini",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        description="Better reasoning, still relatively fast",
        min_ram_gb=8.0,
        min_vram_gb=8.0,
        speed_cpu="Not recommended (OOM)",
        speed_gpu="10-20 sec (6-10 tok/s)",
        quality="Good+",
        recommended_for="T4 GPU, better quality needed",
        quantization_recommended=True,
    ),
    
    "mistral": ModelPreset(
        name="Mistral-7B",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        description="Excellent quality, slower but powerful",
        min_ram_gb=14.0,
        min_vram_gb=14.0,
        speed_cpu="Not supported (OOM)",
        speed_gpu="30-60 sec (3-5 tok/s)",
        quality="Excellent",
        recommended_for="T4 GPU, best quality, default choice",
        quantization_recommended=False,
    ),
    
    "llama13": ModelPreset(
        name="Llama-2-13B",
        model_id="meta-llama/Llama-2-13b-chat-hf",
        description="Highest quality, slowest responses",
        min_ram_gb=26.0,
        min_vram_gb=26.0,
        speed_cpu="Not supported (OOM)",
        speed_gpu="60-120 sec (1-2 tok/s)",
        quality="Best",
        recommended_for="A100 GPU or multiple GPUs, maximum quality",
        quantization_recommended=True,
    ),
}


def get_preset(preset_name: str) -> ModelPreset:
    """
    Get a model preset by name.
    
    Args:
        preset_name: One of 'tiny', 'phi2', 'phi3', 'mistral', 'llama13'
    
    Returns:
        ModelPreset object
    
    Raises:
        ValueError: If preset_name is not recognized
    """
    preset_name = preset_name.lower().strip()
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {available}"
        )
    return PRESETS[preset_name]


def recommend_preset(has_gpu: bool = False, vram_gb: float = 0) -> str:
    """
    Recommend a preset based on available hardware.
    
    Args:
        has_gpu: Whether GPU is available
        vram_gb: Available GPU VRAM in GB (if GPU available)
    
    Returns:
        Recommended preset name
    """
    if not has_gpu:
        # CPU only - use TinyLlama
        return "tiny"
    
    # GPU available - choose based on VRAM
    if vram_gb >= 14:
        return "mistral"  # Default for T4 (16GB)
    elif vram_gb >= 8:
        return "phi3"
    elif vram_gb >= 6:
        return "phi2"
    else:
        return "tiny"


def list_presets() -> None:
    """Print all available presets with their details."""
    print("=" * 80)
    print("AVAILABLE MODEL PRESETS")
    print("=" * 80)
    
    for key, preset in PRESETS.items():
        print(f"\n🔹 {key.upper()}: {preset.name}")
        print(f"   Model: {preset.model_id}")
        print(f"   {preset.description}")
        print(f"   Quality: {preset.quality}")
        print(f"   CPU Speed: {preset.speed_cpu}")
        print(f"   GPU Speed: {preset.speed_gpu}")
        print(f"   RAM needed: {preset.min_ram_gb} GB | VRAM: {preset.min_vram_gb} GB")
        print(f"   Best for: {preset.recommended_for}")
        if preset.quantization_recommended:
            print(f"   💡 Tip: Enable 4-bit quantization for faster inference")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    list_presets()
