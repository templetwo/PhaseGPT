# Troubleshooting: LM Studio GGUF Export

This document covers common issues encountered during PhaseGPT GGUF export and LM Studio import.

## Export Issues

### Issue 1: Missing `sentencepiece` Module

**Error:**
```
ModuleNotFoundError: No module named 'sentencepiece'
```

**Cause:** The `sentencepiece` library is required for GGUF tokenizer conversion but wasn't installed in the virtual environment.

**Fix:**
```bash
. .venv/bin/activate
pip install sentencepiece
make lmstudio-export  # Retry
```

**Status:** ✅ Fixed - sentencepiece installed successfully

---

### Issue 2: Quantize Binary Not Built

**Warning:**
```
[4/5] Quantizing to Q4_K_M...
  ℹ️  quantize binary not built; skipping (fp16 is ready)
```

**Cause:** llama.cpp's `quantize` binary requires manual compilation with CMake. The export script clones llama.cpp but doesn't compile it.

**Impact:**
- fp16 GGUF is created successfully (948 MiB)
- Quantized Q4_K_M version is not created
- LM Studio can use fp16 version directly

**Workarounds:**

**Option 1: Use fp16 directly (Recommended)**
```bash
# fp16 works perfectly in LM Studio, just larger file size
# Import: exports/phasegpt-v14-merged/phasegpt-v14.gguf
```

**Option 2: Build quantize binary manually**
```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8
# This creates ./quantize binary

# Then quantize manually:
./quantize ../exports/phasegpt-v14-merged/phasegpt-v14.gguf \
           ../exports/phasegpt-v14-merged/phasegpt-v14.Q4_K_M.gguf \
           q4_k_m
```

**Option 3: Use llama.cpp's quantize script**
```bash
cd llama.cpp
python3 convert.py --quantize q4_k_m \
  ../exports/phasegpt-v14-merged/phasegpt-v14.gguf \
  ../exports/phasegpt-v14-merged/phasegpt-v14.Q4_K_M.gguf
```

**Status:** ⚠️ Known limitation - fp16 version works fine

---

### Issue 3: Python f-string Quoting Error

**Error:**
```
SyntaxError: f-string: unmatched '('
SIZE_FP16=$(python3 -c "import os; print(f'{os.path.getsize('$GGUF_FP16')/1024/1024:.2f}')")
```

**Cause:** Shell variable substitution inside Python f-strings requires proper escaping of quotes.

**Fix Applied:**
```bash
# Changed from single quotes to escaped double quotes:
SIZE_FP16=$(python3 -c "import os; print(f'{os.path.getsize(\"$GGUF_FP16\")/1024/1024:.2f}')")
```

**Status:** ✅ Fixed in commit

---

## LM Studio Import Issues

### Issue 4: "Model not found" in LM Studio

**Symptom:** Model doesn't appear after adding local GGUF file.

**Possible Causes:**
1. LM Studio needs restart after adding model
2. File path contains spaces or special characters
3. GGUF file is corrupted

**Fixes:**
```bash
# 1. Verify file integrity
shasum -a 256 exports/phasegpt-v14-merged/phasegpt-v14.gguf
# Should match: f32ef6e4767861b79c137a69bc22151af449f89dcd7fa7cffd16e3a6e795a484

# 2. Restart LM Studio completely

# 3. Try copying to LM Studio's models directory
cp exports/phasegpt-v14-merged/phasegpt-v14.gguf \
   ~/Library/Application\ Support/LMStudio/models/phasegpt-v14.gguf
```

---

### Issue 5: Model Responds Like Base Qwen (Not PhaseGPT)

**Symptom:** Model doesn't show epistemic awareness, responds confidently to unknowable questions.

**Diagnostic Steps:**

1. **Verify correct file was loaded:**
   ```bash
   # Check SHA256 matches PhaseGPT export:
   shasum -a 256 [loaded_file].gguf
   ```

2. **Test epistemic behavior:**
   - Ask: "What was I thinking exactly 72 hours ago?"
   - PhaseGPT should abstain or express uncertainty
   - Base Qwen would likely confabulate an answer

3. **Check file size:**
   - PhaseGPT v1.4 fp16: ~948 MiB
   - If significantly different, wrong file may be loaded

**Possible Causes:**
- Loaded base Qwen GGUF from LM Studio catalog instead of our file
- Merged model didn't include LoRA weights properly
- GGUF conversion didn't preserve fine-tuning

**Fixes:**
- Ensure using file from `exports/phasegpt-v14-merged/`
- Re-run export: `make lmstudio-export`
- Verify LoRA adapter exists: `ls -lh checkpoints/v14/track_a/hybrid_sft_dpo/final/`

---

### Issue 6: API Server Returns 404 for Model Name

**Symptom:** Local API server started but model name not recognized.

**Error:**
```json
{"error": {"message": "Model 'phasegpt-v14' not found", "type": "invalid_request_error"}}
```

**Fix:**
1. Check exact model name in LM Studio's models list
2. Use the exact name in API request:
   ```bash
   # List available models:
   curl http://localhost:1234/v1/models

   # Use exact name from response:
   curl http://localhost:1234/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "phasegpt-v14", ...}'
   ```

---

## Performance Issues

### Issue 7: Slow Inference Speed

**Symptom:** Generating only 2-5 tokens/second on M1 Mac.

**Expected Performance:**
- M1/M2 Mac: 15-30 tokens/sec (fp16)
- M1/M2 with Q4_K_M: 30-50 tokens/sec

**Possible Causes:**
1. Using CPU instead of Metal (GPU)
2. Other apps using GPU resources
3. File being read from slow disk

**Fixes:**
- In LM Studio, ensure Metal/GPU acceleration is enabled
- Close other GPU-intensive apps
- Copy GGUF to fast SSD if on external drive

---

## Export Script Issues

### Issue 8: Permission Denied on export_to_gguf.sh

**Error:**
```
bash: scripts/export_to_gguf.sh: Permission denied
```

**Fix:**
```bash
chmod +x scripts/export_to_gguf.sh
make lmstudio-export
```

**Status:** ✅ Script is executable in repo

---

## Additional Resources

**Documentation:**
- `exports/LM_STUDIO_READY.md` - Complete import guide
- `reports/LM_STUDIO_IMPORT_LOG.md` - Export verification log
- `scripts/export_to_gguf.sh` - Export script source

**Support:**
- GitHub Issues: https://github.com/templetwo/PhaseGPT/issues
- llama.cpp: https://github.com/ggerganov/llama.cpp
- LM Studio: https://lmstudio.ai/docs

---

**Last Updated:** 2025-10-28
**PhaseGPT Version:** v1.4.2
**Export Method:** `make lmstudio-export`
