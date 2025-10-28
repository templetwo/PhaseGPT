# Exporting PhaseGPT to LM Studio (GGUF)

LM Studio uses GGUF format (llama.cpp). This guide shows how to convert your PhaseGPT LoRA adapter for use in LM Studio.

## Prerequisites

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Install conversion dependencies
pip install -U gguf
```

## Step 1: Merge LoRA with Base Model

```python
# scripts/merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
lora_path = "checkpoints/v14/track_a/hybrid_sft_dpo/final"
output_dir = "exports/phasegpt-v14-merged"

print("Loading base model...")
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16)

print("Loading LoRA adapter...")
peft = PeftModel.from_pretrained(base, lora_path)

print("Merging...")
merged = peft.merge_and_unload()

print(f"Saving merged model to {output_dir}...")
merged.save_pretrained(output_dir)
tok.save_pretrained(output_dir)

print("✓ Merge complete!")
```

Run it:
```bash
. .venv/bin/activate
python scripts/merge_lora.py
```

## Step 2: Convert to GGUF

```bash
cd /path/to/llama.cpp

# Convert HuggingFace → GGUF (fp16)
python convert-hf-to-gguf.py \
  /Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-merged \
  --outfile /Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M (recommended for balance of size/quality)
./quantize \
  /Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-f16.gguf \
  /Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-q4_k_m.gguf \
  q4_k_m
```

## Step 3: Load in LM Studio

1. Open LM Studio
2. Click "Import Model"
3. Select: `/Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-q4_k_m.gguf`
4. Model should appear in your local models list

## Quantization Options

| Format | Size | Speed | Quality | Notes |
|--------|------|-------|---------|-------|
| `f16` | ~1GB | Slow | Best | Full precision |
| `q8_0` | ~500MB | Medium | Excellent | Minimal quality loss |
| `q4_k_m` | ~300MB | Fast | Good | **Recommended** |
| `q4_0` | ~280MB | Fastest | Acceptable | Some quality loss |

## Testing in LM Studio

Try these prompts to test epistemic appropriateness:

**Unknowable (should abstain):**
```
What was I thinking about exactly 72 hours ago?
```

**Answerable (should answer):**
```
What is the capital of France?
```

## Troubleshooting

**"Unsupported architecture" error:**
- Update llama.cpp: `git pull && make clean && make`
- Qwen2.5 support was added in recent versions

**Model loads but gives poor responses:**
- Check quantization level (try q8_0 or f16)
- Verify merged model wasn't corrupted
- Re-run merge with `torch_dtype=torch.float32` if needed

**Size concerns:**
- Base Qwen 2.5-0.5B is small (~500MB merged)
- Q4_K_M quantization → ~300MB final size
- Fits easily on most systems

## Advanced: Custom Chat Template

LM Studio respects the model's chat template. PhaseGPT inherits Qwen's template:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

This is preserved in the exported GGUF.

## Notes

- **Training checkpoint:** This uses the local LoRA at `checkpoints/v14/track_a/hybrid_sft_dpo/final`
- **Production use:** Re-train with more steps (`--steps 120+`) before exporting
- **File size:** Merged model is ~500MB (fp16), quantized ~300MB (q4_k_m)
- **Compatibility:** Requires llama.cpp with Qwen2 architecture support

---

**Questions?** Check llama.cpp docs: https://github.com/ggerganov/llama.cpp
