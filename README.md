---
title: Zen Max Identity Training
emoji: 🧘
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Train Zen Max identity using QLoRA
---

# Zen Max Identity Training Space

Train Zen Max identity using QLoRA fine-tuning on top of a large MoE base model.

## Features

- **Cloud Training**: All training happens on HuggingFace - no local downloads
- **INT4 Base Model**: K2 already quantized to INT4 (~370GB)
- **QLoRA Efficiency**: LoRA adapters on INT4 model (multi-GPU training)
- **LoRA Adapters**: Only trains adapters (~100MB) not full model
- **Auto-Upload**: Adapters pushed directly to `zenlm/zen-max`

## Base Model

- **Model**: `zenlm/zen-max` (base architecture: Zen Max 671B MoE)
- **Size**: 671B parameters (384 experts, 8 active)
- **BF16 Weights**: ~1.3TB (full precision)
- **INT4 Weights**: ~370GB (quantized on HuggingFace)
- **Training Method**: QLoRA on INT4 model (requires multi-GPU)

## Training Configuration

### Hardware
- GPU: 4x A100 80GB or 8x A100 40GB (provided by Space)
- Model Size: ~370GB (INT4 quantized, 62 shards)
- VRAM Usage: ~500GB total (370GB model + ~130GB activations)
- Training Time: 4-8 hours for 1000 steps

### LoRA Settings
- Rank: 16 (adjustable 4-64)
- Alpha: 32
- Dropout: 0.05
- Target Modules: All attention and MLP layers

### Dataset
- **Source**: `zenlm/zen-identity-dataset`
- **Content**: Zen persona, values, and conversational patterns
- **Size**: Curated high-quality identity examples

## Output

**LoRA Adapters**: Uploaded to `zenlm/zen-max`
- Adapter weights: ~100MB
- Compatible with base K2 model
- Preserves all reasoning capabilities
- Adds Zen identity and values

## Usage After Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (can use 4-bit for inference too)
base_model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-max",  # or the base model path
    device_map="auto",
    load_in_4bit=True
)

# Load Zen adapters
model = PeftModel.from_pretrained(base_model, "zenlm/zen-max")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-max")

# Inference with Zen identity
messages = [{"role": "user", "content": "Tell me about yourself"}]
response = model.chat(tokenizer, messages, thinking_budget=128000)
```

## Why This Approach?

1. **No Downloads**: Train on cloud, never download 1TB model locally
2. **Efficient**: QLoRA uses 4-bit quantization for minimal memory
3. **Modular**: Adapters can be loaded on top of any K2 checkpoint
4. **Practical**: Inference can also use 4-bit for consumer hardware

## Links

- **Base Model**: https://huggingface.co/zenlm/zen-max
- **Output Repo**: https://huggingface.co/zenlm/zen-max
- **Organization**: https://huggingface.co/zenlm
- **Website**: https://zenlm.org

---

**Zen AI**: Clarity Through Intelligence
