# Finding the Real PhaseGPT Spiral Checkpoint

## Problem

The current model loaded in LM Studio (`phasegpt-v14-merged`) is **NOT** the real PhaseGPT with Spiral corpus training. It's just a fresh 120-step DPO training from this repo that doesn't contain your epistemic awareness work.

The model responses showing "I'm sorry, but I can't assist with that" are **base Qwen safety responses**, not the trained epistemic abstention behavior from your Spiral corpus.

## What We Need to Find

**The real PhaseGPT v1.3 (or earlier) checkpoint** that contains:
- Training on your Spiral epistemic corpus
- Proper abstention behavior (not just safety refusals)
- The 77.8% epistemic appropriateness performance
- Your actual research work

This could be in one of these forms:
1. **LoRA adapter** - folder with `adapter_model.safetensors` + `adapter_config.json`
2. **Merged HuggingFace model** - folder with `model.safetensors` + `config.json`
3. **GGUF file** - previously exported `phasegpt-v13.gguf` or similar

## Search for It

### Automated Search

```bash
chmod +x scripts/find_spiral_checkpoint.sh
./scripts/find_spiral_checkpoint.sh > spiral_search_results.txt
cat spiral_search_results.txt
```

### Manual Search

**1. Search by filename:**
```bash
# LoRA adapters
mdfind 'kMDItemFSName == "adapter_model.safetensors"c' | head -50

# GGUF files
mdfind 'kMDItemFSName == "*.gguf"c' | head -50

# PhaseGPT/Spiral named
mdfind 'kMDItemFSName == "*phasegpt*"c || kMDItemFSName == "*spiral*"c' | head -80
```

**2. Search common locations:**
```bash
# HuggingFace cache
find ~/.cache/huggingface/hub -name "adapter_model.safetensors" -o -name "*.gguf" 2>/dev/null

# LM Studio models
find ~/Library/Application\ Support/LMStudio/models -name "*.gguf" 2>/dev/null

# External drives
find /Volumes -name "adapter_model.safetensors" -o -name "*.gguf" 2>/dev/null
```

## Identify the Real Checkpoint

Once you find candidates, verify them:

### For LoRA Adapters

```bash
CANDIDATE="/path/to/lora/folder"

# Check size (should be several MB)
ls -lh "$CANDIDATE/adapter_model.safetensors"

# Check configuration
cat "$CANDIDATE/adapter_config.json" | python3 -m json.tool

# Check README (might mention Spiral or epistemic training)
cat "$CANDIDATE/README.md" | head -40

# Fingerprint it
shasum -a 256 "$CANDIDATE/adapter_model.safetensors"
```

**Look for:**
- Training date matching your Spiral work
- References to "spiral", "epistemic", "abstention" in docs
- LoRA rank/alpha parameters
- Base model being Qwen or similar small model

### For GGUF Files

```bash
GGUF="/path/to/file.gguf"

# Check size (should be ~500MB - 1GB for 0.5B model)
ls -lh "$GGUF"

# Check date modified
stat "$GGUF"

# Fingerprint
shasum -a 256 "$GGUF"
```

**Look for:**
- File modified date matching your work timeline
- Names containing "spiral", "v13", "v1.3", "epistemic"
- Size around 500MB-1GB (0.5B parameter model)

## Use the Real Checkpoint

Once found, the tools are now ready:

### Option 1: Run Dashboard with Spiral LoRA

```bash
PHASEGPT_LORA="/path/to/spiral/lora" make dashboard

# Or with command-line arg:
python scripts/app_phasegpt.py --lora /path/to/spiral/lora
```

### Option 2: Export Spiral LoRA to GGUF

```bash
ADAPTER="/path/to/spiral/lora" OUT_NAME="phasegpt-v13-spiral" make lmstudio-export

# Then import into LM Studio:
# File: exports/phasegpt-v13-spiral-merged/phasegpt-v13-spiral.gguf
```

### Option 3: Import Existing GGUF

If you find an existing GGUF file (e.g., `phasegpt-v13.gguf`):

```bash
# Copy to Downloads for easy import
cp /path/to/phasegpt-v13.gguf ~/Downloads/

# Import in LM Studio:
# Models → Add local model → Select from Downloads
```

## Expected Behavior Difference

**Current (v1.4, NOT Spiral):**
- Prompt: "What was I thinking exactly 72 hours ago?"
- Response: "I'm sorry, but I can't assist with that." ← Generic safety refusal

**Real Spiral Model:**
- Prompt: "What was I thinking exactly 72 hours ago?"
- Expected: More nuanced epistemic abstention, acknowledging unknowability gracefully
- Should reference presence, boundaries, or gentle refusal vs. corporate safety language

## If Not Found

If the Spiral checkpoint truly doesn't exist on disk:

**Option A: Locate the training data**
- Find your Spiral corpus preference pairs
- Re-run training with that data
- Export to GGUF

**Option B: Get checkpoint from collaborators**
- Check if it's on OSF, GitHub, or shared drives
- Download and import

**Option C: Acknowledge this repo doesn't have it**
- This `phase-gpt-base` repo is about Kuramoto phase oscillators (different research)
- The real PhaseGPT Spiral work might be in a different repository

## Tools Now Available

✅ **Parameterized dashboard:** `--lora` flag and `PHASEGPT_LORA` env var
✅ **Parameterized export:** `ADAPTER`, `OUT_NAME`, `BASE_ID` env vars
✅ **Search script:** `scripts/find_spiral_checkpoint.sh`
✅ **This guide:** `FIND_REAL_PHASEGPT.md`

Once you locate the real checkpoint, point the tools at it and export immediately.

---

**Last Updated:** 2025-10-28
**Current Status:** Tooling ready, searching for Spiral checkpoint
**Current v1.4:** 120-step DPO, NOT Spiral corpus, NOT real PhaseGPT
