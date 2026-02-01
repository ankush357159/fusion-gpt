# Model Presets Quick Reference

## Available Presets

veraGPT includes 5 pre-configured model presets for different use cases:

### 1. `tiny` - TinyLlama 1.1B ⚡⚡⚡
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: 1.1 billion parameters
- **RAM Required**: 2-3 GB
- **GPU Speed**: ~15 tokens/second (T4 GPU)
- **CPU Compatible**: ✅ Yes (2-5 min/response)
- **Best For**: Quick responses, simple tasks, CPU-only environments
- **Quality**: Basic - Good for casual chat, simple Q&A

### 2. `phi2` - Microsoft Phi-2 ⚡⚡
- **Model**: `microsoft/phi-2`
- **Size**: 2.7 billion parameters
- **RAM Required**: 6-8 GB
- **GPU Speed**: ~12 tokens/second (T4 GPU)
- **CPU Compatible**: ⚠️ Tight (may OOM on Colab CPU)
- **Best For**: Balanced speed/quality, code generation, reasoning
- **Quality**: Good - Strong reasoning for its size

### 3. `phi3` - Microsoft Phi-3-mini ⚡⚡
- **Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Size**: 3.8 billion parameters
- **RAM Required**: 8-10 GB
- **GPU Speed**: ~10 tokens/second (T4 GPU)
- **CPU Compatible**: ❌ No (requires GPU)
- **Best For**: Better reasoning than Phi-2, still relatively fast
- **Quality**: Very Good - Latest Phi model with improved capabilities

### 4. `mistral` - Mistral 7B Instruct ⚡
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Size**: 7 billion parameters
- **RAM Required**: 14-18 GB
- **GPU Speed**: ~3 tokens/second (T4 GPU)
- **CPU Compatible**: ❌ No (requires GPU, OOM on Colab CPU)
- **Best For**: High-quality responses, complex reasoning, instruction following
- **Quality**: Excellent - Production-grade quality

### 5. `llama13` - Llama-2 13B Chat 🐌
- **Model**: `meta-llama/Llama-2-13b-chat-hf`
- **Size**: 13 billion parameters
- **RAM Required**: 26-32 GB
- **GPU Speed**: ~1-2 tokens/second (T4 GPU)
- **CPU Compatible**: ❌ No (requires high-end GPU)
- **Best For**: Maximum quality, research, complex tasks
- **Quality**: Premium - Best quality, slowest speed
- **Note**: May be slow even on T4 GPU

## Hardware Compatibility Matrix

| Preset | Colab CPU (12GB) | Colab T4 GPU (16GB) | Colab A100 (40GB) |
|--------|------------------|---------------------|-------------------|
| `tiny` | ✅ Works | ✅ Fast (15 tok/s) | ✅ Very Fast |
| `phi2` | ⚠️ May OOM | ✅ Works (12 tok/s) | ✅ Fast |
| `phi3` | ❌ OOM | ✅ Works (10 tok/s) | ✅ Fast |
| `mistral` | ❌ OOM | ✅ Works (3 tok/s) | ✅ Faster |
| `llama13` | ❌ OOM | ⚠️ Slow (1-2 tok/s) | ✅ Works |

## Usage Examples

### In Colab Notebook

```python
# Option 1: Auto-select based on hardware
from model_presets import recommend_preset
import torch

preset = recommend_preset(
    has_gpu=torch.cuda.is_available(),
    vram_gb=16  # T4 GPU
)
# Returns: 'mistral' on GPU, 'tiny' on CPU

# Option 2: Manual selection
SELECTED_PRESET = "phi2"  # Choose any preset

# Load model with preset
from config import Config
from server import ModelServer

config = Config.from_preset(SELECTED_PRESET, enable_quantization=True)
server = ModelServer(config)
server.load()

# Chat
response = server.ask("Explain quantum computing")
print(response)
```

### Via CLI

```bash
# Use preset shortcuts
python src/main.py --model tiny --prompt "Hello!"
python src/main.py --model phi2 --prompt "Explain AI"
python src/main.py --model mistral --prompt "Write Python code"

# With timing
python src/main.py --model tiny --timing --prompt "Quick question"
```

### Via Environment Variable

```bash
# Set preset via environment variable
export LLM_MODEL_PRESET=phi2
python src/main.py --prompt "Hello!"

# Override with CLI
export LLM_MODEL_PRESET=tiny
python src/main.py --model mistral --prompt "Use mistral instead"
```

## Choosing the Right Preset

### For Speed (Quick Responses)
1. **`tiny`** - Fastest, works anywhere (CPU/GPU)
2. **`phi2`** - Good balance, 3-5x faster than mistral
3. **`phi3`** - Slightly slower than phi2, better quality

### For Quality (Best Responses)
1. **`mistral`** - Excellent quality, reasonable speed on GPU
2. **`llama13`** - Best quality, but slow (1-2 tok/s)
3. **`phi3`** - Good compromise between speed and quality

### For CPU Usage
- **Only `tiny` is recommended** - All others will crash with OOM

### For T4 GPU (Colab Free)
- **Recommended**: `phi2`, `phi3`, or `mistral`
- **Avoid**: `llama13` (too slow on T4)

### For Production
- **Recommended**: `mistral` for quality, `phi2` for speed
- **Note**: Production systems should use batching, quantization, and load balancing

## Technical Details

### Quantization
- **Enabled on GPU**: 4-bit quantization (reduces VRAM by ~75%)
- **Disabled on CPU**: Full precision (quantization not supported on CPU)
- **Impact**: 4-bit allows larger models with minimal quality loss

### Model Loading
- **First Load**: 30-60 seconds (downloads + loads weights)
- **Cached**: 10-30 seconds (loads from disk)
- **Memory**: Loaded once, kept in memory for multiple prompts

### Context Length
- All presets support **4K tokens** context (input + output combined)
- Longer conversations may need trimming to fit context window

## Switching Presets

### In Notebook
```python
# Just change the preset and reload
SELECTED_PRESET = "phi2"  # Changed from "tiny"

config = Config.from_preset(SELECTED_PRESET, enable_quantization=True)
server = ModelServer(config)
server.load()  # Loads new model
```

### In CLI
```bash
# Each run can use a different model
python src/main.py --model tiny --prompt "Fast query"
python src/main.py --model mistral --prompt "Quality query"
```

## Custom Configuration

Want to use a different model? Edit `src/model_presets.py`:

```python
PRESETS = {
    # ... existing presets ...
    
    # Add custom preset
    "custom": ModelPreset(
        name="My Custom Model",
        model_id="username/model-name",
        quality="varies",
        speed_gpu="unknown",
        min_vram_gb=8,
    ),
}
```

Then use it:
```python
config = Config.from_preset("custom", enable_quantization=True)
```

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Keep Model Loaded**: Use `ModelServer` to avoid reloading
3. **Enable Quantization**: 4-bit saves memory with minimal quality impact
4. **Shorter Prompts**: Faster generation, less memory
5. **Right Model**: Don't use `llama13` if `phi2` suffices

## Troubleshooting

### "Process killed" or OOM error
- **Solution**: Use smaller preset (`tiny` on CPU, `phi2`/`phi3` on GPU)

### "No CUDA available" error
- **Solution**: Enable GPU in Colab (Runtime → Change runtime type → T4 GPU)

### Slow responses (>2 min)
- **CPU**: Expected, use GPU or `tiny` preset
- **GPU**: Use smaller model (`tiny`, `phi2`) or check GPU utilization

### Model download fails
- **Solution**: Check internet connection, HuggingFace may be down
- **Workaround**: Use cached models or different preset

## More Information

- **Best Practices**: See `MODEL_SELECTION_GUIDE.md`
- **Performance**: See `PERFORMANCE_IMPROVEMENTS.md`
- **Code**: See `src/model_presets.py` for implementation
