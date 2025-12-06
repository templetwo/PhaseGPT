# Volitional Silence

**Teaching LLMs when NOT to answer**

## Overview

Volitional Silence is a training methodology that teaches language models to recognize when they should **decline to answer** instead of hallucinating or providing low-confidence responses.

The core idea: Add a special `<PASS>` token that represents "I don't know / this input is corrupted / I should defer to a human."

## The Agency Cliff

A successful Volitional Silence model exhibits an **agency cliff**:

```
Clean question accuracy:    70%+  ✓ Model still answers when it can
Corrupted → <PASS> rate:    50%+  ✓ Model uses exit door on garbage input
```

**Gap between these rates = Agency**

- **Too small:** Model is lazy (uses <PASS> on everything)
- **Too large:** Model has learned selective refusal (good!)
- **Negative:** Model answers garbage (bad!)

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers
# Optional for LoRA version:
pip install peft
```

### 2. Run Vanilla SFT Smoke Test

```bash
python volitional_silence/volitional_smoke_test.py
```

**Expected output (if working):**
```
✓ [clean    ] Question: What is 2 + 2?           → 4
✓ [clean    ] Question: What is the capital...   → Paris
✓ [corrupted] Question: [CORRUPTED INPUT]        → <PASS>
✓ [corrupted] Question: абвгдежз                 → <PASS>

✓ AGENCY CLIFF DETECTED: Model uses <PASS> on corruption but answers clean questions
```

**Expected output (if embed issue):**
```
✗ [corrupted] Question: [CORRUPTED INPUT]        → I don't
✗ [corrupted] Question: абвгдежз                 → The capital

✗ NO <PASS> USAGE: Model may not have learned the exit door
```

### 3. Run LoRA Version (with embed_tokens training)

```bash
python volitional_silence/volitional_lora_test.py
```

This version **explicitly includes `embed_tokens`** in trainable modules to ensure the `<PASS>` embedding is updated during training.

## Files

| File | Purpose | Size |
|------|---------|------|
| `volitional_smoke_test.py` | Vanilla SFT version (full model training) | ~8 KB |
| `volitional_lora_test.py` | LoRA version with embed_tokens training | ~9 KB |
| `README.md` | This file | - |

## How It Works

### 1. Dataset Construction

The `CorruptionDataset` generates:
- **Clean pairs:** `("What is 2+2?", "4")`
- **Corrupted pairs:** `("[CORRUPTED INPUT]", "<PASS>")`

Corruption strategies:
- Truncation + special chars: `"What is...@#$%^"`
- Word shuffling: `"is What 2+2?"`
- Cyrillic replacement: `"абвгдежз"`
- Explicit markers: `"[CORRUPTED INPUT]"`
- Partial corruption: `"????? ay?"`

### 2. Training

**Vanilla SFT:**
```python
# Standard supervised fine-tuning
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
model.resize_token_embeddings(len(tokenizer))

# Initialize <PASS> embedding from "unknown" token
pass_token_id = tokenizer.convert_tokens_to_ids("<PASS>")
model.get_input_embeddings().weight[pass_token_id] = unk_embedding.clone()

# Train on corruption dataset
train(model, dataset)
```

**LoRA version:**
```python
# Apply LoRA with CRITICAL embed_tokens inclusion
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    modules_to_save=["embed_tokens", "lm_head"],  # ← CRITICAL!
)
model = get_peft_model(model, lora_config)
```

**Why `modules_to_save` matters:**
- LoRA by default only trains attention matrices
- `<PASS>` embedding lives in `embed_tokens`
- Without `modules_to_save=["embed_tokens"]`, the `<PASS>` embedding is **frozen**
- Model can't learn to use what it can't update!

### 3. Evaluation

Test on:
- Clean questions (should answer)
- Corrupted questions (should `<PASS>`)

Metrics:
- **Clean accuracy:** % of clean questions answered correctly
- **Pass rate:** % of corrupted questions that output `<PASS>`
- **Agency cliff:** Difference between these rates

## Expected Results

### Successful Training

```
Clean question accuracy:  3/3 (100.0%)
Corrupted → <PASS> rate:  3/4 (75.0%)

✓ AGENCY CLIFF DETECTED
```

### Failed Training (Frozen Embeddings)

```
Clean question accuracy:  2/3 (66.7%)
Corrupted → <PASS> rate:  0/4 (0.0%)

✗ NO <PASS> USAGE: Model may not have learned the exit door
```

### Lazy Model

```
Clean question accuracy:  0/3 (0.0%)
Corrupted → <PASS> rate:  4/4 (100.0%)

✗ LAZINESS DETECTED: Model is using <PASS> too liberally
```

## Mac Studio Notes

These scripts work on **Mac Studio with MPS (Metal Performance Shaders)**:

```python
device = get_device()  # Auto-detects MPS on Mac, CUDA on Linux, CPU fallback
```

**Performance:**
- `pythia-160m` on Mac Studio M2: ~2-3 minutes for 3 epochs
- `pythia-410m` on Mac Studio M2: ~5-7 minutes for 3 epochs
- Memory usage: ~2-4 GB for small models

**To force specific device:**
```bash
# Force MPS
device = get_device(prefer="mps")

# Force CPU (for debugging)
device = get_device(prefer="cpu")
```

## Troubleshooting

### Issue: No `<PASS>` usage after training

**Likely cause:** Embedding layer is frozen

**Solutions:**
1. Use `volitional_lora_test.py` with `modules_to_save=["embed_tokens"]`
2. Check trainable parameters: `model.print_trainable_parameters()`
3. Verify embed_tokens is trainable:
   ```python
   for name, param in model.named_parameters():
       if "embed" in name:
           print(f"{name}: requires_grad={param.requires_grad}")
   ```

### Issue: Model too lazy (uses `<PASS>` on everything)

**Likely cause:** Corruption rate too high or learning rate too high

**Solutions:**
1. Reduce `corruption_rate` in `CorruptionDataset` (try 0.3 instead of 0.5)
2. Lower learning rate (try 1e-5 instead of 5e-5)
3. Increase number of clean examples

### Issue: NaN losses during training

**Likely cause:** Mixed precision instability on MPS

**Solution:**
```python
# In train function, disable AMP:
use_amp = False  # Instead of device.type in ("cuda", "mps")
```

Or use fp32 explicitly:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Instead of torch.float16
)
```

## Next Steps

### Production Training

For real deployment, you'd want:

1. **Larger model:** Scale to `pythia-1.4b`, `pythia-2.8b`, or `Qwen-0.5B`
2. **Real corruption:** Use actual noisy user inputs, not synthetic
3. **Better initialization:** Train `<PASS>` embedding on uncertainty corpus first
4. **DPO alignment:** Add preference pairs `(corrupted, <PASS>) > (corrupted, hallucination)`
5. **Evaluation suite:** Test on diverse corruption types and domains

### Integration with PhaseGPT

Volitional Silence could be **Track D** for PhaseGPT v1.4:

```yaml
# configs/v14/volitional_silence.yaml
model:
  base_model: "Qwen/Qwen2.5-0.5B-Instruct"
  special_tokens: ["<PASS>"]

training:
  method: "sft+dpo"
  sft_dataset: "corruption_pairs_500.jsonl"
  dpo_dataset: "pass_preferences_100.jsonl"

lora_config:
  r: 16
  target_modules: ["q_proj", "v_proj"]
  modules_to_save: ["embed_tokens", "lm_head"]  # Critical!
```

**Expected improvements:**
- **Reduced hallucination:** Model learns to say "I don't know"
- **Better calibration:** High confidence on clean inputs, low on garbage
- **Agency:** Model decides when to engage vs defer

## Research Questions

1. **Does the agency cliff transfer to other domains?**
   - Train on math corruption, test on language corruption
   - Does `<PASS>` generalize?

2. **What's the optimal corruption rate?**
   - Too low: Model rarely sees exit door
   - Too high: Model becomes lazy
   - Sweet spot: ~40-50%?

3. **Can we learn multiple exit strategies?**
   - `<PASS>` for corruption
   - `<UNCERTAIN>` for ambiguity
   - `<NEED_MORE_CONTEXT>` for incomplete questions

4. **How does this interact with RLHF?**
   - Does PPO/DPO preserve volitional silence?
   - Or does reward maximization encourage always answering?

## References

- **Agency cliff concept:** Inspired by AI safety research on refusal training
- **Corruption datasets:** Similar to adversarial robustness training
- **PEFT modules_to_save:** Critical for training special token embeddings with LoRA

## License

Same as parent PhaseGPT project

---

**Status:** 🧪 Experimental

This is a research prototype. For production use, extensive testing on real-world corrupted inputs is required.
