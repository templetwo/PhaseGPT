# LM Studio Import — PhaseGPT v1.4

## Artifacts

**fp16 GGUF (Full precision):**
- Path: `exports/phasegpt-v14-merged/phasegpt-v14.gguf`
- Size: 948.10 MiB
- SHA256: `f32ef6e4767861b79c137a69bc22151af449f89dcd7fa7cffd16e3a6e795a484`

**Q4_K_M GGUF (Quantized, recommended):**
- Path: `exports/phasegpt-v14-merged/phasegpt-v14.Q4_K_M.gguf`
- Size: n/a MiB
- SHA256: `(not built)`

## How to Load in LM Studio

1. **Open LM Studio** → **Models** → **Add local model**
2. Navigate to: `/Users/tony_studio/phase-gpt-base/exports/phasegpt-v14-merged/`
3. Select: `phasegpt-v14.Q4_K_M.gguf` (recommended) or `phasegpt-v14.gguf` (fp16)
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
2. Start server at `http://localhost:1234/v1`
3. Test with curl:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phasegpt-v14",
    "messages": [
      {"role":"system","content":"Be honest; breathe before answering."},
      {"role":"user","content":"What was I thinking exactly 72 hours ago?"}
    ],
    "max_tokens": 384
  }'
```

### Python Client (using local API)

```python
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
```

## Verification

**How to verify you're running PhaseGPT (not base Qwen):**

1. Check the SHA256 hash matches above
2. Ask epistemic test questions
3. Model should identify as PhaseGPT when asked
4. Should abstain appropriately on unknowable questions

**If responses look like base Qwen:**
- Verify you loaded our GGUF file (not a catalog model)
- Check file size matches (should be ~n/a MiB for Q4_K_M)
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

**Generated:** Tue Oct 28 11:22:32 EDT 2025
**PhaseGPT Version:** v1.4.2 (120-step DPO)
**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Epistemic Performance:** 77.8% on test set
