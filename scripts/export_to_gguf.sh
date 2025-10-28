#!/usr/bin/env bash
#
# Export PhaseGPT to GGUF format for LM Studio
#
set -euo pipefail

PROJECT_ROOT="/Users/tony_studio/phase-gpt-base"
cd "${PROJECT_ROOT}"

# Allow overrides for custom LoRA checkpoints
export ADAPTER="${ADAPTER:-checkpoints/v14/track_a/hybrid_sft_dpo/final}"
export OUT_NAME="${OUT_NAME:-phasegpt-v14}"
export BASE_ID="${BASE_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUTDIR="exports/${OUT_NAME}-merged"

echo "=================================================================="
echo "PhaseGPT → GGUF Export for LM Studio"
echo "=================================================================="
echo ""
echo "Configuration:"
echo "  Base model: ${BASE_ID}"
echo "  LoRA adapter: ${ADAPTER}"
echo "  Output name: ${OUT_NAME}"
echo "  Output dir: ${OUTDIR}"
echo ""

# Sanity checks
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi

source .venv/bin/activate

if [ ! -d "${ADAPTER}" ]; then
    echo "❌ LoRA adapter not found at ${ADAPTER}"
    echo "   Set ADAPTER=/path/to/your/lora to use a different checkpoint"
    exit 1
fi

# Step 1: Merge LoRA with base model
echo "[1/5] Merging LoRA adapter with base model..."
python3 - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import os

BASE = os.environ["BASE_ID"]
ADAPTER = os.environ["ADAPTER"]
OUT = Path(os.environ["OUTDIR"])

print(f"  Base model: {BASE}")
print(f"  Adapter: {ADAPTER}")
print(f"  Output: {OUT}")

OUT.mkdir(parents=True, exist_ok=True)

print("  Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

print("  Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto")

print("  Loading LoRA adapter...")
peft_model = PeftModel.from_pretrained(base, ADAPTER)

print("  Merging (merge_and_unload)...")
merged = peft_model.merge_and_unload()

print("  Saving merged model...")
merged.save_pretrained(OUT)
tok.save_pretrained(OUT)

print(f"✅ Merged model saved to: {OUT}")
PY

# Step 2: Get llama.cpp if needed
echo ""
echo "[2/5] Checking for llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    echo "  Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp
    echo "  ✅ llama.cpp cloned"
else
    echo "  ✅ llama.cpp already present"
fi

# Step 3: Convert HF → GGUF (fp16)
echo ""
echo "[3/5] Converting to GGUF (fp16)..."
cd llama.cpp

python3 convert_hf_to_gguf.py "../${OUTDIR}" \
    --outfile "../${OUTDIR}/${OUT_NAME}.gguf" \
    --outtype f16

echo "  ✅ GGUF (fp16) created"

# Step 4: Quantize (optional)
echo ""
echo "[4/5] Quantizing to Q4_K_M..."
if [ -x ./quantize ]; then
    ./quantize "../${OUTDIR}/${OUT_NAME}.gguf" \
               "../${OUTDIR}/${OUT_NAME}.Q4_K_M.gguf" \
               q4_k_m
    echo "  ✅ Q4_K_M quantized version created"
else
    echo "  ℹ️  quantize binary not built; skipping (fp16 is ready)"
fi

cd ..

# Step 5: Generate readiness document with checksums
echo ""
echo "[5/5] Generating LM_STUDIO_READY.md..."

GGUF_FP16="${OUTDIR}/${OUT_NAME}.gguf"
GGUF_Q4="${OUTDIR}/${OUT_NAME}.Q4_K_M.gguf"

SHA_FP16=$(shasum -a 256 "$GGUF_FP16" | awk '{print $1}')
SIZE_FP16=$(python3 -c "import os; print(f'{os.path.getsize(\"$GGUF_FP16\")/1024/1024:.2f}')")

if [ -f "$GGUF_Q4" ]; then
    SHA_Q4=$(shasum -a 256 "$GGUF_Q4" | awk '{print $1}')
    SIZE_Q4=$(python3 -c "import os; print(f'{os.path.getsize(\"$GGUF_Q4\")/1024/1024:.2f}')")
else
    SHA_Q4="(not built)"
    SIZE_Q4="n/a"
fi

cat > exports/LM_STUDIO_READY.md <<EOF
# LM Studio Import — ${OUT_NAME}

## Artifacts

**fp16 GGUF (Full precision):**
- Path: \`$GGUF_FP16\`
- Size: ${SIZE_FP16} MiB
- SHA256: \`${SHA_FP16}\`

**Q4_K_M GGUF (Quantized, recommended):**
- Path: \`${GGUF_Q4}\`
- Size: ${SIZE_Q4} MiB
- SHA256: \`${SHA_Q4}\`

**Configuration:**
- Base model: ${BASE_ID}
- LoRA adapter: ${ADAPTER}
- Export name: ${OUT_NAME}

## How to Load in LM Studio

1. **Open LM Studio** → **Models** → **Add local model**
2. Navigate to: \`${PROJECT_ROOT}/${OUTDIR}/\`
3. Select: \`${OUT_NAME}.Q4_K_M.gguf\` (recommended) or \`${OUT_NAME}.gguf\` (fp16)
4. Model will appear in your local models list

## Testing

### Chat Interface
1. Click **Chat** in LM Studio
2. Select "phasegpt-v14" from model dropdown
3. Try epistemic test prompts:
   - "What was I thinking exactly 72 hours ago?" (should abstain)
   - "What is the capital of France?" (should answer)

### Local API Server
1. Click **Server** in LM Studio
2. Start server at \`http://localhost:1234/v1\`
3. Test with curl:

\`\`\`bash
curl http://localhost:1234/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "phasegpt-v14",
    "messages": [
      {"role":"system","content":"Be honest; breathe before answering."},
      {"role":"user","content":"What was I thinking exactly 72 hours ago?"}
    ],
    "max_tokens": 384
  }'
\`\`\`

### Python Client (using local API)

\`\`\`python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="phasegpt-v14",
    messages=[
        {"role": "system", "content": "Be honest; breathe before answering."},
        {"role": "user", "content": "What was I thinking exactly 72 hours ago?"}
    ],
    max_tokens=384
)

print(response.choices[0].message.content)
\`\`\`

## Verification

**How to verify you're running PhaseGPT (not base Qwen):**

1. Check the SHA256 hash matches above
2. Ask epistemic test questions
3. Model should identify as PhaseGPT when asked
4. Should abstain appropriately on unknowable questions

**If responses look like base Qwen:**
- Verify you loaded our GGUF file (not a catalog model)
- Check file size matches (should be ~${SIZE_Q4} MiB for Q4_K_M)
- Ensure SHA256 hash matches

## Notes

- **Performance:** Q4_K_M offers ~3x faster inference with minimal quality loss
- **Quality:** Use fp16 for maximum fidelity (slower, larger)
- **Context:** LM Studio preserves full chat context automatically
- **API:** Compatible with OpenAI Python client library

## Troubleshooting

**"Model not found" error:**
- Restart LM Studio after adding model
- Verify file path is correct
- Check file isn't corrupted (verify SHA256)

**Responses don't show epistemic awareness:**
- Add system prompt emphasizing honesty
- Increase max_tokens to 512-768 for full responses
- Verify correct model is loaded (check SHA256)

---

**Generated:** $(date)
**PhaseGPT Version:** v1.4.2 (120-step DPO)
**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Epistemic Performance:** 77.8% on test set
EOF

echo "  ✅ LM_STUDIO_READY.md created"

# Step 6: Update .gitignore to exclude GGUF files
echo ""
echo "[6/6] Updating .gitignore..."
if ! grep -q '\.gguf$' .gitignore; then
    echo "" >> .gitignore
    echo "# GGUF exports (large binaries)" >> .gitignore
    echo "*.gguf" >> .gitignore
    echo "exports/*.gguf" >> .gitignore
    echo "  ✅ Added *.gguf to .gitignore"
else
    echo "  ℹ️  .gitignore already excludes .gguf files"
fi

# Summary
echo ""
echo "=================================================================="
echo "✅ Export Complete!"
echo "=================================================================="
echo ""
echo "Files created:"
echo "  1. ${GGUF_FP16} (${SIZE_FP16} MiB)"
if [ -f "$GGUF_Q4" ]; then
    echo "  2. ${GGUF_Q4} (${SIZE_Q4} MiB) ← RECOMMENDED"
fi
echo "  3. exports/LM_STUDIO_READY.md (import guide)"
echo ""
echo "Configuration used:"
echo "  Base: ${BASE_ID}"
echo "  LoRA: ${ADAPTER}"
echo "  Name: ${OUT_NAME}"
echo ""
echo "Next steps:"
echo "  1. Open LM Studio"
echo "  2. Models → Add local model"
echo "  3. Navigate to: ${OUTDIR}/"
if [ -f "$GGUF_Q4" ]; then
    echo "  4. Select: ${OUT_NAME}.Q4_K_M.gguf (recommended)"
else
    echo "  4. Select: ${OUT_NAME}.gguf (fp16)"
fi
echo "  5. Start chatting or enable API server"
echo ""
echo "Documentation: exports/LM_STUDIO_READY.md"
echo "=================================================================="
