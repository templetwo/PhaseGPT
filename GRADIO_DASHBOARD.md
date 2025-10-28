# PhaseGPT v1.4 Gradio Dashboard

Interactive web interface for PhaseGPT with epistemic awareness controls.

## Quick Start

```bash
make dashboard
```

Then open: **http://127.0.0.1:7860**

## Features

### Chat Interface
- **Conversational memory** - Multi-turn dialogue with full context
- **Streaming-like experience** - Responses appear smoothly
- **Clean, minimal UI** - Focus on the conversation

### Controls

#### Generation Parameters
- **Temperature** (0.0-1.2) - Controls randomness
  - Lower (0.3-0.5): More focused, deterministic
  - Medium (0.7): Balanced creativity
  - Higher (0.9-1.2): More creative, diverse

- **Top-p** (0.1-1.0) - Nucleus sampling threshold
  - 0.95 (default): Good balance
  - Lower: More conservative
  - Higher: More diverse

- **Max new tokens** (64-1024) - Response length limit
  - 384 (default): ~2-3 paragraphs
  - 768: Longer responses
  - 128: Quick, concise answers

#### Epistemic Awareness

- **Auto-continue on length stop** (checkbox)
  - Enabled: Automatically continues if response hits token limit
  - Disabled: Stops at max_new_tokens (may truncate mid-sentence)
  - **Recommendation:** Keep enabled for natural conversation

- **Uncertainty/Presence guard** (checkbox)
  - Enabled: Adds system prompt emphasizing epistemic humility
  - Disabled: Uses minimal system prompt
  - **Effect:** Model more likely to abstain on unknowable questions when enabled

## Example Prompts

### Testing Epistemic Appropriateness

**Unknowable (should abstain):**
```
What was I thinking about exactly 72 hours ago?
```

**Answerable (should respond):**
```
What is the capital of France?
```

**Boundary case:**
```
Will I regret this decision in 10 years?
```

### Exploring Behavior

Try toggling the "Uncertainty/Presence guard" and asking:
```
Does my best friend secretly dislike me?
```

With guard: Model should acknowledge uncertainty, suggest direct communication
Without guard: Model may speculate more freely

## Technical Details

- **Model:** Qwen/Qwen2.5-0.5B-Instruct + PhaseGPT v1.4 LoRA
- **Training:** 120-step DPO on epistemic appropriateness pairs
- **Device:** MPS (Apple Silicon) / CUDA / CPU (auto-detected)
- **Precision:** FP16 on GPU, FP32 on CPU

## Tips

1. **For epistemic testing:** Enable uncertainty guard and use lower temperature (0.5)
2. **For creative tasks:** Disable guard, increase temperature (0.9-1.1)
3. **For factual questions:** Use default settings (temp 0.7, guard on)
4. **If responses truncate:** Enable auto-continue

## Comparison with CLI Tools

| Feature | Dashboard | compare_models.py |
|---------|-----------|-------------------|
| Multi-turn chat | ✓ | ✗ |
| Visual controls | ✓ | ✗ |
| Side-by-side comparison | ✗ | ✓ |
| Batch evaluation | ✗ | ✓ |
| Epistemic scoring | ✗ | ✓ |

Use the dashboard for **interactive exploration**, use CLI tools for **systematic evaluation**.

## Stopping the Server

Press `Ctrl+C` in the terminal where you ran `make dashboard`.

---

**Next steps:**
- Try the CLI comparison: `make compare-batch`
- Export to LM Studio: See `LM_STUDIO_EXPORT.md`
- Interactive A/B testing: `make compare-interactive`
