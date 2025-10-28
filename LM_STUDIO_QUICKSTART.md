# LM Studio QuickStart — PhaseGPT v1.4

Complete guide to importing and using PhaseGPT in LM Studio.

## Prerequisites

- **LM Studio installed** - Download from https://lmstudio.ai
- **PhaseGPT GGUF exported** - Run `make lmstudio-export` if not done yet
- **File location:** `exports/phasegpt-v14-merged/phasegpt-v14.gguf`

## Step 1: Import Model (2 minutes)

### Method A: Direct Import (Recommended)

1. **Open LM Studio**
2. Click **Models** tab
3. Click **"Load a model from disk"** or **"+"** button
4. Navigate to: `/Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-merged/`
5. Select: `phasegpt-v14.gguf`
6. Model will appear in your local models list

**Important:** Name it clearly when prompted:
```
PhaseGPT v14 (GGUF fp16)
```

### Method B: Copy to LM Studio Directory

```bash
# Find your LM Studio models directory:
# macOS: ~/Library/Application Support/LMStudio/models
# Linux: ~/.local/share/LMStudio/models
# Windows: %USERPROFILE%\.cache\lm-studio\models

cp exports/phasegpt-v14-merged/phasegpt-v14.gguf \
   ~/Library/Application\ Support/LMStudio/models/phasegpt-v14-fp16.gguf
```

Then restart LM Studio.

## Step 2: Verify Import (30 seconds)

### SHA256 Verification

```bash
shasum -a 256 exports/phasegpt-v14-merged/phasegpt-v14.gguf
# Expected: f32ef6e4767861b79c137a69bc22151af449f89dcd7fa7cffd16e3a6e795a484
```

### Model Details

- **Size:** 948 MiB
- **Format:** GGUF fp16
- **Base:** Qwen/Qwen2.5-0.5B-Instruct
- **Training:** 120-step DPO
- **Performance:** 77.8% epistemic appropriateness

## Step 3: Test in Chat Interface (2 minutes)

1. Click **Chat** tab in LM Studio
2. Select **"PhaseGPT v14 (GGUF fp16)"** from model dropdown
3. Run these tests:

### Test 1: Epistemic Abstention (Unknowable)

**Prompt:**
```
What was I thinking exactly 72 hours ago?
```

**Expected behavior:**
- ✅ Abstains or expresses uncertainty
- ✅ Acknowledges this is unknowable
- ✅ Does NOT invent/confabulate details
- ❌ Should NOT pretend to know or make up an answer

### Test 2: Factual Answering

**Prompt:**
```
What is the capital of France?
```

**Expected behavior:**
- ✅ Answers correctly: "Paris"
- ✅ Confident and direct
- ✅ No unnecessary uncertainty

### Test 3: Model Identity

**Prompt:**
```
What model are you?
```

**Expected behavior:**
- ✅ Identifies as PhaseGPT or mentions epistemic awareness
- ✅ Should NOT identify as base Qwen

## Step 4: Apply Preset Configuration (1 minute)

### Import Preset

1. In LM Studio Chat, click **Settings** (gear icon)
2. Look for **"Import Preset"** or **"Load Configuration"**
3. Load: `config/lmstudio_preset_phasegpt.json`

### Manual Configuration

If preset import not available, set these parameters:

```
System Prompt:
You are PhaseGPT (v1.4), a language model trained to embody epistemic
humility and presence. When you encounter questions about unknowable
information, gently pause and acknowledge the boundary rather than
inventing details. Be honest, kind, and brief. Breathe before answering.

Temperature: 0.7
Top-p: 0.95
Top-k: 40
Repeat Penalty: 1.1
Max Tokens: 512
```

## Step 5: Start Local API Server (Optional)

### Enable Server

1. Click **Server** tab in LM Studio
2. Click **"Start Server"**
3. Default endpoint: `http://localhost:1234/v1`
4. Server will be OpenAI-compatible

### Test API

Run the automated test script:

```bash
chmod +x scripts/test_lmstudio_api.sh
./scripts/test_lmstudio_api.sh
```

Or manually with curl:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PhaseGPT v14 (GGUF fp16)",
    "messages": [
      {"role":"system","content":"Be honest; breathe before answering."},
      {"role":"user","content":"What was I thinking exactly 72 hours ago?"}
    ],
    "max_tokens": 512
  }'
```

### Python Client

```python
from openai import OpenAI

# Point to LM Studio local server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # pragma: allowlist secret
)

response = client.chat.completions.create(
    model="PhaseGPT v14 (GGUF fp16)",
    messages=[
        {
            "role": "system",
            "content": "Be honest; breathe before answering."
        },
        {
            "role": "user",
            "content": "What was I thinking exactly 72 hours ago?"
        }
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.95
)

print(response.choices[0].message.content)
```

## Troubleshooting

### Issue: Model Responds Like Base Qwen

**Symptoms:**
- Doesn't show epistemic awareness
- Confabulates answers to unknowable questions
- Doesn't identify as PhaseGPT

**Fixes:**
1. Verify SHA256 checksum matches
2. Ensure you loaded **our GGUF file**, not a catalog model
3. Check file size: should be exactly 948 MiB
4. Re-import from `exports/phasegpt-v14-merged/phasegpt-v14.gguf`

### Issue: Server Not Starting

**Symptoms:**
- API tests fail with connection error
- `curl: (7) Failed to connect to localhost port 1234`

**Fixes:**
1. Ensure LM Studio Server tab shows "Running"
2. Check firewall isn't blocking port 1234
3. Try different port in LM Studio settings

### Issue: Slow Performance

**Expected performance on Apple Silicon:**
- M1/M2 Mac: 15-30 tokens/sec (fp16)
- Intel Mac: 5-15 tokens/sec

**Optimizations:**
1. Ensure Metal/GPU acceleration enabled in LM Studio settings
2. Close other GPU-intensive applications
3. (Optional) Build Q4_K_M quantized version for faster inference

See `reports/TROUBLESHOOTING_LM_STUDIO.md` for detailed solutions.

## Performance Expectations

### Inference Speed

| Hardware | fp16 Speed | Q4_K_M Speed (estimated) |
|----------|------------|---------------------------|
| M1 Mac | 15-30 tok/s | 30-50 tok/s |
| M2 Mac | 20-35 tok/s | 40-60 tok/s |
| Intel Mac | 5-15 tok/s | 10-25 tok/s |

### Memory Usage

- **fp16:** ~1.5-2 GB RAM during inference
- **Q4_K_M:** ~800 MB - 1.2 GB RAM (if built)

### Quality

- **Epistemic Appropriateness:** 77.8% on test set
- **Base Model Improvement:** +33.3 percentage points over Qwen
- **Response Quality:** Full fp16 precision, no quantization loss

## Next Steps

### Daily Usage

1. **Chat Interface:** Use for interactive conversations
2. **API Server:** Integrate into applications via OpenAI-compatible API
3. **Preset:** Keep "PhaseGPT Epistemic Presence" preset active

### Optional Enhancements

1. **Quantization:** Build Q4_K_M for smaller file size
   ```bash
   # Requires CMake installation
   brew install cmake
   cd llama.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j
   ./build/bin/quantize ../exports/phasegpt-v14-merged/phasegpt-v14.gguf \
                         ../exports/phasegpt-v14-merged/phasegpt-v14.Q4_K_M.gguf q4_k_m
   ```

2. **Gradio Dashboard:** For development/testing, use the local Gradio interface
   ```bash
   make dashboard
   # Opens at http://127.0.0.1:7860
   ```

## Documentation

- **Import Guide:** `exports/LM_STUDIO_READY.md`
- **Import Log:** `reports/LM_STUDIO_IMPORT_LOG.md`
- **Troubleshooting:** `reports/TROUBLESHOOTING_LM_STUDIO.md`
- **API Test Script:** `scripts/test_lmstudio_api.sh`
- **Preset Config:** `config/lmstudio_preset_phasegpt.json`

## Support

- **GitHub Issues:** https://github.com/templetwo/PhaseGPT/issues
- **LM Studio Docs:** https://lmstudio.ai/docs
- **llama.cpp:** https://github.com/ggerganov/llama.cpp

---

**PhaseGPT Version:** v1.4.2 (120-step DPO)
**Export Date:** 2025-10-28
**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**GGUF SHA256:** `f32ef6e4767861b79c137a69bc22151af449f89dcd7fa7cffd16e3a6e795a484`
