# 🎯 Quick Start Guide - Model Selection

## 1️⃣ Open Colab Notebook
Open `vera_gpt.ipynb` in Google Colab

## 2️⃣ Choose Your Hardware

### Option A: CPU Only (Free)
```
Runtime → Change runtime type → None (CPU)
```
**✅ Works with**: `tiny` preset only  
**⚡ Speed**: 2-5 minutes per response

### Option B: GPU (Recommended)
```
Runtime → Change runtime type → T4 GPU
```
**✅ Works with**: All presets  
**⚡ Speed**: 5-60 seconds per response (depending on model)

## 3️⃣ Run Setup Cells
Run cells 1-4 to install dependencies and clone the repo.

## 4️⃣ Select Your Model

Cell 5 will show you all available models. Cell 6 auto-recommends the best one for your hardware.

**Quick Selection Guide**:

| If You Want... | Choose This |
|----------------|-------------|
| 🚀 Fastest speed | `tiny` (15 tok/s on GPU, works on CPU) |
| ⚖️ Balanced | `phi2` (12 tok/s) or `phi3` (10 tok/s) |
| 🎯 Best quality | `mistral` (3 tok/s) - default |
| 🏆 Maximum quality | `llama13` (1-2 tok/s, slow) |
| 💻 CPU only | `tiny` - **only option for CPU** |

In Cell 7, change this line:
```python
SELECTED_PRESET = "tiny"  # Change to: "phi2", "phi3", "mistral", or "llama13"
```

## 5️⃣ Load Model
Run Cell 8. This takes 30-60 seconds. The model loads **once** and stays in memory.

## 6️⃣ Chat!
Run Cell 9 to chat. Each response takes 5-60 seconds depending on model.

**Pro tip**: Ask multiple questions without reloading!
```python
server.ask("What is Python?")       # Fast!
server.ask("Explain decorators")    # Fast!
server.ask("Write a web scraper")   # Fast!
```

## 🔄 Want to Try a Different Model?
Run the optional cell (after Cell 9):
```python
SELECTED_PRESET = "phi2"  # Switch from "tiny" to "phi2"
# Run cell to reload
```

## 🛠️ Troubleshooting

### ❌ "Process killed" or OOM Error
**Problem**: Model too big for available RAM  
**Solution**: 
- On CPU: Use `tiny` preset only
- On GPU: Use `tiny`, `phi2`, or `phi3` instead of `mistral`

### ❌ "CUDA out of memory"
**Problem**: Model too big for GPU VRAM  
**Solution**: Use smaller preset or enable quantization
```python
config = Config.from_preset("phi2", enable_quantization=True)
```

### 🐌 Very Slow Responses (>2 minutes)
**Problem**: Running on CPU or model too large  
**Solution**: 
- Enable GPU: `Runtime → Change runtime type → T4 GPU`
- Use smaller model: Switch to `tiny` or `phi2`

### ⚠️ "No module named 'model_presets'"
**Problem**: Working directory not set correctly  
**Solution**: Run this in a cell:
```python
%cd /content/fusion-gpt/veraGPT
import sys
sys.path.insert(0, '/content/fusion-gpt/veraGPT/src')
```

## 📊 Performance Comparison

Tested on Google Colab:

| Preset | CPU (12GB RAM) | T4 GPU (16GB VRAM) | Quality |
|--------|----------------|---------------------|---------|
| `tiny` | 2-5 min/response | 5-10 sec (15 tok/s) | ⭐⭐ Basic |
| `phi2` | ❌ OOM | 8-12 sec (12 tok/s) | ⭐⭐⭐ Good |
| `phi3` | ❌ OOM | 10-15 sec (10 tok/s) | ⭐⭐⭐⭐ Very Good |
| `mistral` | ❌ OOM | 30-40 sec (3 tok/s) | ⭐⭐⭐⭐⭐ Excellent |
| `llama13` | ❌ OOM | 60-90 sec (1-2 tok/s) | ⭐⭐⭐⭐⭐ Premium |

## 🎓 CLI Usage (Alternative)

If you prefer command line:

```bash
# Quick test
python src/main.py --model tiny --prompt "Hello!"

# With timing stats
python src/main.py --model phi2 --timing --prompt "Explain AI"

# Interactive mode
python src/main.py --model mistral

# With quantization (saves memory)
python src/main.py --model mistral --quant 4
```

## 📚 More Help

- **Detailed preset info**: See `MODEL_PRESETS.md`
- **Model selection guide**: See `MODEL_SELECTION_GUIDE.md`
- **Performance tips**: See `PERFORMANCE_IMPROVEMENTS.md`
- **Full README**: See `README.md`

## 💡 Pro Tips

1. **Keep model loaded**: Use `server.ask()` multiple times without reloading
2. **Start small**: Test with `tiny` first, then upgrade to `mistral` if needed
3. **Enable GPU**: Always use GPU for better speed (except for `tiny`)
4. **Use quantization**: Add `enable_quantization=True` to save memory
5. **Check hardware**: Run cell 5 to see auto-recommendations

---

**Need help?** Check the troubleshooting section in `vera_gpt.ipynb` (last cell) or see the full documentation in `MODEL_PRESETS.md`.

Happy chatting! 🚀
