## veraGPT - Multi-Model LLM Chatbot

A modular, production-ready chatbot with **5 pre-configured model presets** (TinyLlama, Phi-2, Phi-3, Mistral-7B, Llama-2-13B). Supports quantization, LoRA fine-tuning, streaming inference, and persistent chat history.

🚀 **New**: Easy model selection with presets - choose speed or quality based on your hardware!

### Project Structure

```
veraGPT/
├── main.py                 # CLI entry-point & argument parser
├── config.py               # Centralised configuration (all tunables)
├── model_presets.py        # 🆕 Pre-configured model presets (5 models)
├── server.py               # 🆕 Persistent model server (fast multi-prompt)
├── logger.py               # Logging setup (file + console)
├── device_manager.py       # GPU / MPS / CPU selection & dtype logic
├── model_loader.py         # Download, cache, quantize, load model
├── prompt_formatter.py     # Chat-template formatting
├── inference_engine.py     # Blocking + streaming text generation
├── chat_history.py         # In-memory + JSON-persisted chat sessions
├── chatbot.py              # Interactive REPL tying everything together
└── requirements.txt        # Python dependencies
```

### Why this layout?

| File                  | Single Responsibility                                                |
| --------------------- | -------------------------------------------------------------------- |
| `config.py`           | Every tunable lives here – change one place, affects everything      |
| `device_manager.py`   | All hardware logic isolated; easy to add TPU support later           |
| `model_loader.py`     | Knows how to load / quantize / attach LoRA – nothing else            |
| `prompt_formatter.py` | Owns the Mistral chat template – swap for Llama template in one file |
| `inference_engine.py` | Owns generation logic – blocking and streaming                       |
| `chat_history.py`     | Owns conversation state and persistence                              |
| `chatbot.py`          | Orchestrator – wires modules together, owns the REPL                 |
| `main.py`             | CLI glue – parses args, delegates to chatbot or one-shot mode        |

### Quick Start

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For **CUDA GPU** support, install the correct PyTorch wheel first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For **4-bit / 8-bit quantization** (CUDA only), also install:

```bash
pip install bitsandbytes
```

#### 2. Choose a Model Preset

veraGPT includes 5 pre-configured models optimized for different hardware:

| Preset | Model | Size | Speed (T4 GPU) | CPU Support | Best For |
|--------|-------|------|----------------|-------------|----------|
| `tiny` | TinyLlama 1.1B | 2-3 GB RAM | ⚡⚡⚡ 15 tok/s | ✅ Yes | Fast responses, CPU usage |
| `phi2` | Microsoft Phi-2 | 6-8 GB RAM | ⚡⚡ 12 tok/s | ⚠️ Tight | Balanced speed/quality |
| `phi3` | Microsoft Phi-3-mini | 8-10 GB RAM | ⚡⚡ 10 tok/s | ❌ No | Better reasoning |
| `mistral` | Mistral 7B | 14-18 GB RAM | ⚡ 3 tok/s | ❌ No | High quality (default) |
| `llama13` | Llama-2 13B | 26-32 GB RAM | 🐌 1-2 tok/s | ❌ No | Maximum quality |

**Quick Selection**:
- **CPU only?** Use `tiny` preset
- **T4 GPU (Colab free)?** Use `phi2`, `phi3`, or `mistral`
- **Want speed?** Use `tiny` or `phi2`
- **Want quality?** Use `mistral` or `llama13`

#### 3. Run the chatbot

```bash
# Using presets (recommended)
python main.py --model tiny        # TinyLlama (fastest)
python main.py --model phi2        # Phi-2 (balanced)
python main.py --model mistral     # Mistral-7B (default, best quality)

# Interactive mode with timing stats
python main.py --model tiny --timing

# Single prompt
python main.py --model phi2 --prompt "Explain quantum computing"

# With quantization (GPU only, saves VRAM)
python main.py --model mistral --quant 4

# Custom system prompt
python main.py --model phi3 --system "You are a Python expert."

# Via environment variable
export LLM_MODEL_PRESET=tiny
python main.py
```

#### 4. Persistent Server (Fast Multi-Prompt)

For multiple prompts, use the persistent server to avoid reloading the model:

```bash
# Start Python interpreter
python

# Load model once
from config import Config
from server import ModelServer

config = Config.from_preset("phi2", enable_quantization=True)
server = ModelServer(config)
server.load()  # Takes 30-60s

# Ask multiple questions (fast - no reload!)
server.ask("What is Python?")      # ~5-10 seconds
server.ask("Explain decorators")   # ~5-10 seconds  
server.ask("Write a sorting algo") # ~5-10 seconds

# Or use interactive mode
server.interactive()  # REPL mode
```

**Performance**: After initial load, subsequent prompts are 10-100x faster (no model reload overhead).

#### 5. Full CLI reference

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `mistral` | Model preset: `tiny`, `phi2`, `phi3`, `mistral`, `llama13` OR full HF model id |
| `--local-path` | — | Load from a local directory instead of Hub |
| `--device` | `auto` | `auto \| cuda \| mps \| cpu` |
| `--quant` | — | `4` or `8` (requires bitsandbytes, CUDA only) |
| `--timing` | off | 🆕 Show generation timing stats (tokens/sec, latency) |
| `--max-tokens` | `512` | Max new tokens per response |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling cutoff |
| `--top-k` | `50` | Top-k pool size |
| `--greedy` | off | Greedy decoding (ignores temp/top-p/top-k) |
| `--system` | — | System prompt |
| `--session-id` | new UUID | Resume a named session |
| `--no-stream` | off | Disable token streaming |
| `--prompt` | — | One-shot prompt then exit |
| `--log-level` | `INFO` | `DEBUG \| INFO \| WARNING \| ERROR` |

### Environment Variables

All CLI flags can also be set via environment variables (CLI takes precedence):

| Variable             | Equivalent flag          | Example |
| -------------------- | ------------------------ | ------- |
| `LLM_MODEL_PRESET`   | 🆕 `--model` (preset)    | `tiny`, `phi2`, `mistral` |
| `LLM_MODEL_PATH`     | `--model` (full path)    | `mistralai/Mistral-7B-Instruct-v0.2` |
| `LLM_MAX_NEW_TOKENS` | `--max-tokens`           | `512` |
| `LLM_TEMPERATURE`    | `--temperature`          | `0.7` |
| `LLM_QUANT_ENABLED`  | Set to `1` for `--quant` | `1` |
| `LLM_QUANT_BITS`     | `4` or `8`               | `4` |
| `LLM_DEVICE`         | `--device`               | `cuda`, `cpu` |
| `LLM_LOG_LEVEL`      | `--log-level`            | `INFO` |

**Example**:
```bash
export LLM_MODEL_PRESET=phi2
export LLM_QUANT_ENABLED=1
export LLM_QUANT_BITS=4
python main.py  # Uses Phi-2 with 4-bit quantization
```

### In-Chat Commands

While in the interactive REPL, type any of these:

| Command            | Effect                     |
| ------------------ | -------------------------- |
| `/help`            | Show all commands          |
| `/quit` or `/exit` | Save history and exit      |
| `/clear`           | Wipe current conversation  |
| `/save`            | Force-save history to disk |
| `/history`         | Print full conversation    |
| `/stream`          | Toggle streaming on / off  |

### Configuration Deep-Dive (`config.py`)

Everything is a Python dataclass – import and override programmatically:

```python
from config import Config

# Method 1: Use preset (recommended)
cfg = Config.from_preset("phi2", enable_quantization=True)

# Method 2: From environment variables
cfg = Config.from_env()

# Method 3: Manual override
cfg = Config()
cfg.model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
cfg.generation.temperature = 0.3
cfg.generation.max_new_tokens = 1024
cfg.quantization.enabled = True
cfg.quantization.load_in_bits = 4
```

### Model Presets Deep-Dive

See all available presets programmatically:

```python
from model_presets import PRESETS, list_presets, recommend_preset
import torch

# List all presets with details
list_presets()

# Auto-recommend based on hardware
preset = recommend_preset(
    has_gpu=torch.cuda.is_available(),
    vram_gb=16  # T4 GPU
)
print(f"Recommended: {preset}")  # Returns 'mistral' on GPU, 'tiny' on CPU

# Get specific preset details
from model_presets import get_preset
phi2 = get_preset("phi2")
print(f"Model: {phi2.model_id}")
print(f"Min VRAM: {phi2.min_vram_gb} GB")
```

### Quantization

Requires a **CUDA GPU** and the `bitsandbytes` package.

| Bits        | VRAM saving | Speed impact    |
| ----------- | ----------- | --------------- |
| 8-bit       | ~25%        | Minimal         |
| 4-bit (NF4) | ~50%        | Small (~10-15%) |

### LoRA Fine-Tuning

Enable LoRA in `config.py`:

```python
cfg.lora.enabled = True
cfg.lora.rank = 16
cfg.lora.target_modules = ["q_proj", "v_proj"]
```

Then install `peft`:

```bash
pip install peft
```

The adapter is attached automatically during `ModelLoader.load()`.

### Extending the Project

| Goal | Where to change |
| --- | --- |
| Add new model preset | `model_presets.py` → Add to `PRESETS` dict |
| Swap to a custom model | Use `--model <full-hf-path>` OR add preset |
| Change prompt template | `prompt_formatter.py` → Update chat template |
| Add a REST API | Create `api.py`, import `InferenceEngine` or `ModelServer` |
| Add RAG (retrieval) | Create `retriever.py`, inject retrieved context into `PromptFormatter` |
| Change default system prompt | `prompt_formatter.py` → `DEFAULT_SYSTEM_PROMPT` |
| Persist to a database | Subclass or extend `ChatHistory` |
| Multi-GPU tensor parallelism | Already handled: `device_map="auto"` in `model_loader.py` |

### Documentation

- **[MODEL_PRESETS.md](MODEL_PRESETS.md)**: Detailed guide to all 5 model presets
- **[MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md)**: How to choose the right model
- **[PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md)**: Performance optimization details

### Hardware Requirements

| Preset | Min VRAM/RAM | CPU Support | Colab Free (T4) | Recommended For |
|--------|--------------|-------------|-----------------|-----------------|
| `tiny` | 2-3 GB | ✅ Yes | ✅ Fast (15 tok/s) | CPU usage, quick tests |
| `phi2` | 6-8 GB | ⚠️ Tight | ✅ Fast (12 tok/s) | Balanced performance |
| `phi3` | 8-10 GB | ❌ No | ✅ Good (10 tok/s) | Better reasoning |
| `mistral` | 14-18 GB | ❌ No | ✅ Works (3 tok/s) | Production quality |
| `llama13` | 26-32 GB | ❌ No | ⚠️ Slow (1-2 tok/s) | Maximum quality |

**Quantization Impact** (GPU only):
- **FP16** (no quant): Full VRAM requirement
- **8-bit**: ~25% VRAM savings, minimal quality loss
- **4-bit (NF4)**: ~50% VRAM savings, small quality loss (~10%)

**Colab Quick Guide**:
- **CPU Runtime** (12 GB RAM): Only `tiny` preset works
- **T4 GPU** (16 GB VRAM): All presets except `llama13` (too slow)
- **A100 GPU** (40 GB VRAM): All presets work great

### Google Colab Notebook

The easiest way to get started is using the included Colab notebook:

1. Open `vera_gpt.ipynb` in Google Colab
2. Run cells 1-4 to clone repo and install dependencies
3. Cell 5-6 auto-detect hardware and recommend a model preset
4. Cell 7 lets you manually select a preset (`tiny`, `phi2`, `phi3`, `mistral`)
5. Cell 8 loads the model (30-60 seconds)
6. Cell 9-10 chat with the model

**Features**:
- ✅ Auto-detects CPU vs GPU runtime
- ✅ Recommends appropriate model based on hardware
- ✅ Persistent model server (no reload between prompts)
- ✅ Easy preset switching
- ✅ Troubleshooting guide for OOM errors

**Performance Comparison** (from notebook):
```
TinyLlama (CPU):  2-5 min/response  ❌ Slow
TinyLlama (T4):   5-10 sec/response ✅ Fast (15 tok/s)
Phi-2 (T4):       8-12 sec/response ✅ Fast (12 tok/s)
Mistral (T4):     30-40 sec/response ⚡ Medium (3 tok/s)
Mistral (CPU):    ❌ OOM/Killed
```

### License

- This project is as-is for educational and research purposes.
- The model weights are gated by Mistral AI's license [Mistral-7B model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
