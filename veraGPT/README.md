## Mistral-7B-Instruct Chatbot

A modular, production-ready chatbot built on top of **mistralai/Mistral-7B-Instruct-v0.2** with support for quantization, LoRA fine-tuning, streaming inference, and persistent chat history.

### Project Structure

```
llm_project/
├── main.py                 # CLI entry-point & argument parser
├── config.py               # Centralised configuration (all tunables)
├── logger.py               # Logging setup (file + console)
├── device_manager.py       # GPU / MPS / CPU selection & dtype logic
├── model_loader.py         # Download, cache, quantize, load model
├── prompt_formatter.py     # Mistral chat-template formatting
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

#### 2. Run the chatbot

```bash
# Interactive mode (streams tokens by default)
python main.py

# With 4-bit quantization (saves ~12 GB VRAM)
python main.py --quant 4

# Single prompt – no REPL
python main.py --prompt "Explain transformers in simple terms."

# Custom system prompt
python main.py --system "You are a Shakespearean poet. Reply only in iambic pentameter."
```

#### 3. Full CLI reference

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `mistralai/Mistral-7B-Instruct-v0.2` | HF model id or local path |
| `--local-path` | — | Load from a local directory instead of Hub |
| `--device` | `auto` | `auto \| cuda \| mps \| cpu` |
| `--quant` | — | `4` or `8` (requires bitsandbytes) |
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

| Variable             | Equivalent flag          |
| -------------------- | ------------------------ |
| `LLM_MODEL_PATH`     | `--model`                |
| `LLM_MAX_NEW_TOKENS` | `--max-tokens`           |
| `LLM_TEMPERATURE`    | `--temperature`          |
| `LLM_QUANT_ENABLED`  | Set to `1` for `--quant` |
| `LLM_QUANT_BITS`     | `4` or `8`               |
| `LLM_DEVICE`         | `--device`               |
| `LLM_LOG_LEVEL`      | `--log-level`            |

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

cfg = Config()
cfg.generation.temperature = 0.3
cfg.generation.max_new_tokens = 1024
cfg.quantization.enabled = True
cfg.quantization.load_in_bits = 4
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
| Swap to a different model (e.g. Llama-3) | `config.py` → `ModelConfig.model_name_or_path` + update `prompt_formatter.py` chat template |
| Add a REST API | Create `api.py`, import `InferenceEngine`, expose endpoints |
| Add RAG (retrieval) | Create `retriever.py`, inject retrieved context into `PromptFormatter` |
| Change default system prompt | `prompt_formatter.py` → `DEFAULT_SYSTEM_PROMPT` |
| Persist to a database | Subclass or extend `ChatHistory` |
| Multi-GPU tensor parallelism | Already handled: `device_map="auto"` in `model_loader.py` |

### Hardware Requirements

| Setup           | Min VRAM       | Recommended            |
| --------------- | -------------- | ---------------------- |
| FP16 (no quant) | ~14 GB         | 16 GB+ GPU             |
| 8-bit quantized | ~10 GB         | 12 GB+ GPU             |
| 4-bit quantized | ~6 GB          | 8 GB+ GPU              |
| CPU only        | N/A (uses RAM) | 32 GB+ RAM (very slow) |

### License

- This project is as-is for educational and research purposes.
- The model weights are gated by Mistral AI's license [Mistral-7B model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
