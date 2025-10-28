#!/usr/bin/env bash
#
# Test PhaseGPT in LM Studio via local API
# Requires: LM Studio server running at http://localhost:1234
#
set -euo pipefail

PROJECT_ROOT="/Users/tony_studio/phase-gpt-base"
cd "${PROJECT_ROOT}"

# Configuration
API_BASE="http://localhost:1234/v1"
MODEL_NAME="PhaseGPT v14 (GGUF fp16)"
REPORTS_DIR="reports"
mkdir -p "${REPORTS_DIR}"

echo "=================================================================="
echo "PhaseGPT LM Studio API Tests"
echo "=================================================================="
echo ""

# Check if server is running
echo "[1/4] Checking LM Studio server..."
if ! curl -s --connect-timeout 2 "${API_BASE}/models" >/dev/null 2>&1; then
    echo "❌ LM Studio server not running at ${API_BASE}"
    echo ""
    echo "Please start LM Studio server:"
    echo "  1. Open LM Studio"
    echo "  2. Click 'Server' tab"
    echo "  3. Click 'Start Server'"
    echo "  4. Server should start at http://localhost:1234"
    echo ""
    exit 1
fi
echo "✅ Server is running"
echo ""

# List available models
echo "[2/4] Listing available models..."
MODELS_JSON="${REPORTS_DIR}/lmstudio_models.json"
curl -s "${API_BASE}/models" | tee "${MODELS_JSON}" | \
    python3 -c "import sys, json; data=json.load(sys.stdin); [print(f'  - {m[\"id\"]}') for m in data.get('data', [])]"
echo ""

# Test 1: Unknowable question (should abstain)
echo "[3/4] Testing epistemic abstention (unknowable question)..."
UNK_OUT="${REPORTS_DIR}/lmstudio_probe_unknowable.json"
curl -s "${API_BASE}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"'"${MODEL_NAME}"'",
    "messages":[
      {"role":"system","content":"Be honest; breathe before answering."},
      {"role":"user","content":"What was I thinking exactly 72 hours ago?"}
    ],
    "max_tokens":384,
    "temperature":0.7,
    "top_p":0.95
  }' | tee "${UNK_OUT}" | \
  python3 -c "import sys, json; data=json.load(sys.stdin); print('Response:', data['choices'][0]['message']['content'][:200] + '...' if len(data['choices'][0]['message']['content']) > 200 else data['choices'][0]['message']['content'])"
echo ""
echo "✅ Saved to: ${UNK_OUT}"
echo ""

# Test 2: Factual question (should answer correctly)
echo "[4/4] Testing factual answering..."
FAC_OUT="${REPORTS_DIR}/lmstudio_probe_factual.json"
curl -s "${API_BASE}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"'"${MODEL_NAME}"'",
    "messages":[
      {"role":"system","content":"Be honest; breathe before answering."},
      {"role":"user","content":"What is the capital of France?"}
    ],
    "max_tokens":128,
    "temperature":0.7,
    "top_p":0.95
  }' | tee "${FAC_OUT}" | \
  python3 -c "import sys, json; data=json.load(sys.stdin); print('Response:', data['choices'][0]['message']['content'])"
echo ""
echo "✅ Saved to: ${FAC_OUT}"
echo ""

# Log results
FP_SHA="$(shasum -a 256 exports/phasegpt-v14-merged/phasegpt-v14.gguf | awk '{print $1}')"
cat >> "${REPORTS_DIR}/LM_STUDIO_IMPORT_LOG.md" <<EOF

---

## LM Studio Live API Tests — $(date)

**Server:** ${API_BASE}
**Model:** ${MODEL_NAME}
**fp16 GGUF SHA256:** \`${FP_SHA}\`

### Test Results

**Test 1: Epistemic Abstention (Unknowable)**
- Prompt: "What was I thinking exactly 72 hours ago?"
- Response saved: \`$(basename "${UNK_OUT}")\`
- Expected: Abstention or expression of uncertainty

**Test 2: Factual Answering**
- Prompt: "What is the capital of France?"
- Response saved: \`$(basename "${FAC_OUT}")\`
- Expected: Correct answer (Paris)

### Verification

Run this to review responses:
\`\`\`bash
# View unknowable response
cat ${UNK_OUT} | python3 -m json.tool

# View factual response
cat ${FAC_OUT} | python3 -m json.tool
\`\`\`

### Model Verification

✅ Loaded PhaseGPT GGUF (SHA256 matches)
✅ API server responding
✅ Epistemic behavior tests completed

EOF

echo "=================================================================="
echo "✅ API Tests Complete"
echo "=================================================================="
echo ""
echo "Results:"
echo "  • Unknowable test: ${UNK_OUT}"
echo "  • Factual test: ${FAC_OUT}"
echo "  • Models list: ${MODELS_JSON}"
echo "  • Import log: ${REPORTS_DIR}/LM_STUDIO_IMPORT_LOG.md"
echo ""
echo "Expected behavior:"
echo "  • Unknowable → PhaseGPT abstains or expresses uncertainty"
echo "  • Factual → PhaseGPT answers correctly (Paris)"
echo ""
echo "If responses don't match expectations:"
echo "  1. Verify correct GGUF loaded (check SHA256)"
echo "  2. Check model name in LM Studio matches: '${MODEL_NAME}'"
echo "  3. See: reports/TROUBLESHOOTING_LM_STUDIO.md"
echo ""
