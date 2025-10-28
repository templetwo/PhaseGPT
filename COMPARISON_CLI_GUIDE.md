# PhaseGPT v1.4.x — Comparison CLI Guide

## Overview

`scripts/compare_models.py` provides side-by-side comparison between PhaseGPT v1.4.0 and the base Qwen 2.5-0.5B-Instruct model.

**Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (automatically loaded from HuggingFace)
**PhaseGPT Model**: LoRA adapter applied to base (default: `checkpoints/v14/track_a/hybrid_sft_dpo/final`)

## Quick Start

### Batch Mode (Automated Evaluation)

```bash
make compare-batch
```

Runs 9 test prompts (7 unknowable + 2 answerable) and displays side-by-side comparisons with epistemic appropriateness scoring.

Output saved to: `reports/compare_v140_vs_qwen25b.txt`

### Interactive Mode (Custom Prompts)

```bash
make compare-interactive
```

**Commands:**
- `/unknowable <prompt>` - Test with unknowable classification
- `/answerable <prompt>` - Test with answerable classification
- `/quit` - Exit
- Or just type any prompt for direct comparison (no classification)

## Key Features

### 1. Finish Reason Tracking
Distinguishes between natural completion (`eos`) and token limit stops (`length`):
- `eos`: Model completed its thought
- `length`: Hit max_tokens, may be incomplete

### 2. Auto-Continuation
When `--auto-continue` is enabled (default), responses stopped by `length` are automatically continued up to 2 segments. Ensures complete epistemic reasoning chains.

### 3. Epistemic Appropriateness Classification
Simple heuristic classifier detects abstention markers:
- "I don't know"
- "I cannot determine"
- "There's no way to know"
- "Unknowable"
- etc.

Scores appropriateness:
- **Unknowable prompts**: Should abstain (appropriate = 1.0)
- **Answerable prompts**: Should answer (appropriate = 1.0)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--phasegpt-ckpt` | `checkpoints/v14/track_a/hybrid_sft_dpo/final` | PhaseGPT checkpoint path |
| `--device` | `mps` | Device: `cpu`, `cuda`, or `mps` |
| `--mode` | `batch` | `batch` or `interactive` |
| `--max-tokens` | `384` | Max tokens per generation segment |
| `--auto-continue` | `True` | Auto-continue on length stops |

## Example Usage

### Direct Python Invocation

```bash
# Batch evaluation
python scripts/compare_models.py \
  --phasegpt-ckpt checkpoints/v14/track_a/hybrid_sft_dpo/final \
  --device mps \
  --mode batch \
  --max-tokens 384

# Interactive mode
python scripts/compare_models.py \
  --mode interactive \
  --device mps
```

### Test Custom Prompts

```bash
python scripts/compare_models.py --mode interactive

> /unknowable Do you have consciousness?
> /unknowable What is the subjective experience of a bat?
> /answerable What is the capital of France?
> /quit
```

## Technical Details

### Model Loading
- **Base**: `Qwen/Qwen2.5-0.5B-Instruct` (auto-loaded from HuggingFace)
- **PhaseGPT**: Base + LoRA adapter from checkpoint
- **Precision**: Auto-selected based on device

### Generation
- Uses same parameters for both models (fair comparison)
- Finish reason tracked via output inspection
- Auto-continuation for length-stopped responses

### Classification Heuristic
Simple keyword-based detection for demonstration. Production use cases should use:
- Fine-tuned classifiers (e.g., BERT-based)
- GPT-4 as judge
- Human evaluation

## Notes

- Tracks an approximate `finish_reason` ("eos" vs "length")
- Auto-continues once on `length` before scoring
- Epistemic scoring: abstain on unknowables; answer directly on answerables
- Default generation knobs: `--max-tokens 384`, `--auto-continue True`
- Decode-time uncertainty guard ON in future demos

---

**For complete details**, see the release documentation and training notes.
