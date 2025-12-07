# Architectural Adaptation Guide for Volitional Silence

**Based on: "Architectural Adaptation of LoRA Configurations for Special Token Embedding Training: A Comparative Analysis of Qwen2.5 and Pythia Architectures"**

---

## Executive Summary

When training special tokens like `<PASS>` with LoRA, the **critical failure mode** is using the wrong module names for `modules_to_save`. This causes the embedding layer to remain frozen, preventing the model from learning the new token.

**The Problem:**
- Standard LoRA freezes ALL base model parameters
- New tokens (e.g., `<PASS>`) are initialized as random noise
- If embeddings stay frozen, the model cannot learn what `<PASS>` means
- **Solution:** Use `modules_to_save` to unfreeze embeddings

**The Trap:**
- Different architectures use different module names
- Qwen/Llama: `embed_tokens` + `lm_head`
- Pythia/GPT-NeoX: `embed_in` + `embed_out`
- Using the wrong names = silent failure (no training, no error)

---

## Architecture Comparison Table

| Component | Qwen2.5/Llama | Pythia/GPT-NeoX | Notes |
|-----------|---------------|-----------------|-------|
| **Input Embedding** | `embed_tokens` | `embed_in` | Where token IDs → vectors |
| **Output Projection** | `lm_head` | `embed_out` | Where vectors → logits |
| **Root Module** | `model` | `gpt_neox` | Container for transformer |
| **Attention Q/K/V** | `q_proj`, `k_proj`, `v_proj` | `query_key_value` (fused) | Separate vs fused |
| **Attention Output** | `o_proj` | `dense` | Output projection |
| **MLP Expansion** | `gate_proj`, `up_proj` | `dense_h_to_4h` | Feed-forward up |
| **MLP Contraction** | `down_proj` | `dense_4h_to_h` | Feed-forward down |
| **Embedding Tying** | Untied (`tie_word_embeddings: false`) | Untied | Both require dual-target |

---

## The Dual-Target Requirement

Modern architectures use **untied embeddings**:

```
Input:  Token ID → embed_tokens → Vector (Qwen) or embed_in (Pythia)
Output: Vector → lm_head → Logits (Qwen) or embed_out (Pythia)
```

**These are SEPARATE weight matrices** in memory.

**Implication:**
- Training only `embed_tokens`/`embed_in` = model can READ `<PASS>` but NOT WRITE it
- Training only `lm_head`/`embed_out` = model can WRITE `<PASS>` but NOT READ it
- **Must train BOTH** for bidirectional token competency

**Common Error:**
```python
# ❌ WRONG - Only trains input embedding
modules_to_save=["embed_tokens"]

# ✅ CORRECT - Trains both input and output
modules_to_save=["embed_tokens", "lm_head"]
```

---

## Correct Configurations

### Qwen2.5 / Llama Family

```python
from peft import LoraConfig, TaskType

qwen_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Target attention + MLP for LoRA adaptation
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
        "gate_proj", "up_proj", "down_proj"         # MLP
    ],

    # CRITICAL: Unfreeze BOTH embedding layers
    modules_to_save=[
        "embed_tokens",  # Input: token ID → vector
        "lm_head"        # Output: vector → logits
    ],

    bias="none",
    inference_mode=False
)
```

**Applies to:**
- Qwen, Qwen2, Qwen2.5 (all sizes)
- Llama, Llama 2, Llama 3
- Mistral, Mixtral
- Phi-2, Phi-3

### Pythia / GPT-NeoX Family

```python
from peft import LoraConfig, TaskType

pythia_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Target attention + MLP (NeoX-specific names!)
    target_modules=[
        "query_key_value",   # Fused QKV (not q_proj/k_proj/v_proj!)
        "dense",             # Attention output projection
        "dense_h_to_4h",     # MLP expansion
        "dense_4h_to_h"      # MLP contraction
    ],

    # CRITICAL: Unfreeze BOTH embedding layers (NeoX names!)
    modules_to_save=[
        "embed_in",   # Input: token ID → vector
        "embed_out"   # Output: vector → logits
    ],

    bias="none",
    inference_mode=False
)
```

**Applies to:**
- Pythia (all sizes: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b)
- GPT-NeoX-20B
- StableLM (some variants)

---

## Architecture Detection Strategy

### Automatic Detection

```python
def detect_architecture(model):
    """Detect architecture from model config."""
    model_type = model.config.model_type.lower()

    # Qwen/Llama lineage
    if any(x in model_type for x in ['qwen', 'llama', 'mistral', 'phi']):
        return {
            "embed_input": "embed_tokens",
            "embed_output": "lm_head",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }

    # Pythia/NeoX lineage
    elif any(x in model_type for x in ['gpt_neox', 'pythia']):
        return {
            "embed_input": "embed_in",
            "embed_output": "embed_out",
            "target_modules": ["query_key_value", "dense"]
        }

    else:
        raise ValueError(f"Unknown architecture: {model_type}")
```

### Manual Verification

```python
# Check what embedding modules exist
for name, module in model.named_modules():
    if 'embed' in name.lower():
        print(f"Found: {name}")

# Expected output (Qwen):
# Found: model.embed_tokens
# Found: ...

# Expected output (Pythia):
# Found: gpt_neox.embed_in
# Found: embed_out
```

---

## Common Errors and Fixes

### Error 1: No `<PASS>` Usage After Training

**Symptom:**
```
Clean question accuracy:  70%
Corrupted → <PASS> rate:  0%   ← Should be 30%+

✗ NO <PASS> USAGE: Model may not have learned the exit door
```

**Diagnosis:**
```python
model.print_trainable_parameters()
# If output shows only ~1-2% trainable (just LoRA weights)
# Then embeddings are FROZEN
```

**Root Cause:** Wrong `modules_to_save` for the architecture

**Fix:**
```python
# Check model type
print(model.config.model_type)  # e.g., "qwen2" or "gpt_neox"

# Use correct config:
# Qwen: modules_to_save=["embed_tokens", "lm_head"]
# Pythia: modules_to_save=["embed_in", "embed_out"]
```

### Error 2: Size Mismatch on Load

**Symptom:**
```
RuntimeError: size mismatch for base_model.model.model.embed_tokens.weight:
copying a param with shape torch.Size([152000, 3584]) from checkpoint,
the shape in current model is torch.Size([151936, 3584])
```

**Root Cause:** Base model loaded without resizing vocabulary

**Fix:**
```python
# CORRECT ORDER:
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))  # ← CRITICAL BEFORE load_adapter

model = PeftModel.from_pretrained(model, adapter_path)
```

### Error 3: AttributeError on Pythia

**Symptom:**
```
AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'model'
```

**Root Cause:** Trying to access `model.model.embed_tokens` (Qwen path) on Pythia

**Fix:**
```python
# Qwen access pattern:
model.model.embed_tokens  # ✓

# Pythia access pattern:
model.gpt_neox.embed_in   # ✓
```

### Error 4: vLLM Serving Failure

**Symptom:**
```
ValueError: There is no module or parameter named 'layers'
```

**Root Cause:** vLLM doesn't support `modules_to_save` adapters directly

**Fix:**
```python
# Merge adapter into base model before serving
model = model.merge_and_unload()
model.save_pretrained("merged-model")
tokenizer.save_pretrained("merged-model")

# Then serve merged model with vLLM
```

---

## Storage Implications

### Standard LoRA (No Special Tokens)

```
Checkpoint size: ~50-200 MB
Contents: Low-rank matrices (A, B) for targeted layers
```

### LoRA with `modules_to_save`

```
Checkpoint size: ~2-3 GB (for 7B models)
Contents:
  - Low-rank matrices (~50 MB)
  - Full embedding layer (~1 GB)
  - Full output head (~1 GB)
```

**Example (Qwen2.5-7B):**
- Vocab size: 152,000
- Hidden size: 3,584
- Precision: bfloat16 (2 bytes)
- `embed_tokens`: 152,000 × 3,584 × 2 = 1.09 GB
- `lm_head`: 152,000 × 3,584 × 2 = 1.09 GB
- **Total:** ~2.2 GB + LoRA weights

**Mitigation:** Use `trainable_token_indices` for surgical training (advanced)

---

## Verification Checklist

Before training:
- [ ] Checked `model.config.model_type`
- [ ] Used architecture-specific `modules_to_save`
- [ ] Called `model.resize_token_embeddings(len(tokenizer))`
- [ ] Initialized new token embeddings (e.g., from "unknown")

After applying LoRA:
- [ ] Called `model.print_trainable_parameters()`
- [ ] Verified trainable params > 1B (for 7B models)
- [ ] Checked embedding layers are trainable:
  ```python
  for name, p in model.named_parameters():
      if 'embed' in name and p.requires_grad:
          print(f"✓ {name}")
  ```

After training:
- [ ] Evaluated `<PASS>` usage rate > 30%
- [ ] Verified model can both READ and WRITE `<PASS>`
- [ ] Tested checkpoint save/load cycle

---

## Advanced: `trainable_token_indices`

For users who cannot afford 2GB+ checkpoints:

```python
# Instead of unfreezing entire embedding layer:
modules_to_save=["embed_tokens", "lm_head"]  # ← 2GB checkpoint

# Train only specific token indices:
trainable_token_indices={
    "embed_tokens": [152064, 152065],  # Just the new tokens
    "lm_head": [152064, 152065]
}  # ← ~kB checkpoint
```

**Pros:**
- Tiny checkpoints
- Only new tokens trained

**Cons:**
- Limited backend support (vLLM, TGI)
- May hurt semantic integration
- Requires merging before serving

---

## References

### Official Documentation
- PEFT LoRA Guide: https://huggingface.co/docs/peft/main/en/task_guides/lora
- Qwen2.5 Docs: https://qwenlm.github.io/
- Pythia Paper: https://arxiv.org/abs/2304.01373

### Key GitHub Issues
- vLLM + modules_to_save: https://github.com/vllm-project/vllm/issues/9280
- Qwen2.5-VL embed_tokens: https://github.com/QwenLM/Qwen3-VL/issues/1402
- Unsloth resizing: Stack Overflow thread on PEFT model resizing

### Model Configs
- Qwen2.5-7B: `tie_word_embeddings: false` (confirmed)
- Pythia models: Typically untied embeddings

---

## Quick Reference

| If your model is... | Use `modules_to_save=` |
|---------------------|----------------------|
| Qwen, Qwen2, Qwen2.5 | `["embed_tokens", "lm_head"]` |
| Llama, Llama 2, Llama 3 | `["embed_tokens", "lm_head"]` |
| Mistral, Mixtral | `["embed_tokens", "lm_head"]` |
| Phi-2, Phi-3 | `["embed_tokens", "lm_head"]` |
| Pythia (all sizes) | `["embed_in", "embed_out"]` |
| GPT-NeoX-20B | `["embed_in", "embed_out"]` |
| Unknown | Check `model.config.model_type` and inspect with `model.named_modules()` |

---

**Last Updated:** 2025-12-06
**Based on:** Architectural Adaptation Analysis by templetwo
**Implementation:** Volitional Silence v2 (architecture-aware)
