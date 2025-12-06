# Mac Studio Training Guide for PhaseGPT

**Hardware:** Mac Studio (M1/M2 Ultra) with 36GB Unified Memory
**GPU Backend:** MPS (Metal Performance Shaders)
**Status:** ✅ Fully Supported (no code changes needed!)

---

## 🎉 **Great News: MPS Already Supported!**

PhaseGPT's training code **already has full MPS support built in**. You don't need to modify anything - just use `--device mps` or `--device auto` and it will work!

---

## 🚀 **Quick Start**

### **1. Verify PyTorch MPS Support**

```bash
python3 - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("\n✅ MPS is ready!")
    print("You can use --device mps or --device auto for training")
else:
    print("\n❌ MPS not available")
    print("You may need to upgrade PyTorch: pip install --upgrade torch")
PY
```

**Expected Output:**
```
PyTorch version: 2.1.0+ (or newer)
MPS available: True
MPS built: True

✅ MPS is ready!
You can use --device mps or --device auto for training
```

### **2. Run Track A Training with Auto-Detection**

```bash
# PhaseGPT automatically detects MPS on Mac
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device auto

# Or explicitly specify MPS
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device mps
```

---

## 💾 **Memory Optimization for 36GB Unified Memory**

Your Mac Studio has **36GB unified memory** (shared between CPU and GPU). Here's how to optimize:

### **Recommended Batch Sizes**

| Model Size | LoRA Rank | Batch Size | Gradient Accum | Effective Batch | Expected Memory |
|------------|-----------|------------|----------------|-----------------|-----------------|
| Qwen 0.5B  | r=16      | 8          | 4              | 32              | ~8-12 GB        |
| Qwen 0.5B  | r=24      | 4          | 8              | 32              | ~10-14 GB       |
| Qwen 1.5B  | r=16      | 4          | 8              | 32              | ~18-24 GB       |
| Qwen 1.5B  | r=24      | 2          | 16             | 32              | ~20-28 GB       |
| Pythia 2.8B| r=16      | 4          | 8              | 32              | ~22-30 GB       |

**Your 36GB is perfect for:**
- ✅ Qwen 0.5B (Track A & B) - plenty of headroom
- ✅ Qwen 1.5B (Track C) - comfortable fit with gradient checkpointing
- ✅ Pythia 2.8B - tight but doable with optimizations

### **Memory-Saving Flags**

Edit your config YAML to include:

```yaml
training:
  batch_size: 4                    # Start conservative
  gradient_accumulation_steps: 8   # Effective batch = 32

model:
  use_gradient_checkpointing: true  # Trade compute for memory

# Optional: Use fp32 (MPS mixed precision is still maturing)
training:
  fp16: false
  bf16: false
```

---

## ⚡ **Performance Expectations**

### **Training Speed (Approximate)**

MPS is **slower than NVIDIA GPUs** but still very usable for research:

| Model      | Hardware        | Tokens/sec | Time for 3 Epochs (100 pairs) |
|------------|-----------------|------------|-------------------------------|
| Qwen 0.5B  | Mac Studio M2   | ~500-800   | ~2-3 hours                    |
| Qwen 0.5B  | A100 (40GB)     | ~2000-3000 | ~30-45 minutes                |
| Qwen 1.5B  | Mac Studio M2   | ~200-400   | ~6-10 hours                   |
| Qwen 1.5B  | A100 (40GB)     | ~1000-1500 | ~1-2 hours                    |

**Bottom line:** You'll wait a few hours instead of minutes, but it's totally viable for Track A/B experiments.

---

## 🐛 **Known MPS Quirks & Workarounds**

### **1. Mixed Precision (AMP) Issues**

**Problem:** MPS mixed precision (`torch.autocast("mps")`) can be unstable

**Solution:** Use fp32 training (it's slower but stable)

```yaml
# In your config YAML
training:
  fp16: false
  bf16: false
  # MPS will use fp32 - no autocast
```

### **2. Some Ops Fall Back to CPU**

**Problem:** Rare ops might fall back to CPU with a warning

**Solution:** Usually harmless, but if slow:
```python
# In Python, set this before training:
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### **3. Out of Memory Errors**

**Solution:** Reduce batch size or enable gradient checkpointing

```bash
# If training crashes with OOM:
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device mps \
  --batch-size 2 \
  --gradient-accum 16
```

### **4. NumPy/PyTorch Version Mismatches**

**Problem:** Sometimes MPS requires specific PyTorch versions

**Solution:** Use PyTorch 2.1+ and NumPy <2.0

```bash
pip install torch>=2.1.0 'numpy<2.0'
```

---

## 📝 **Track A Training on Mac Studio**

### **Step-by-Step Workflow**

```bash
# 1. Verify MPS is available
python3 scripts/verify_mps.py  # (script below)

# 2. Generate or verify dataset (already done if you synced from Linux)
make validate-data

# 3. Run a quick test (10 steps to verify everything works)
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device mps \
  --max-steps 10 \
  --output-dir checkpoints/test_mps

# 4. Check the output - should see "Using device: mps"
# Look for: "Training completed successfully" in logs

# 5. Run full Track A training
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device auto \
  --output-dir checkpoints/v14/track_a/mac_run_$(date +%Y%m%d_%H%M%S)

# 6. Monitor with Activity Monitor
# Open: Applications > Utilities > Activity Monitor
# Watch: "GPU" tab to see Metal utilization
```

### **Expected Log Output**

```
2025-10-27 14:30:00 [INFO] Using device: mps
2025-10-27 14:30:01 [INFO] Trainer initialized:
2025-10-27 14:30:01 [INFO]   Device: mps
2025-10-27 14:30:01 [INFO]   Model type: Phase-coupled
2025-10-27 14:30:01 [INFO]   Total parameters: 523,456,789
2025-10-27 14:30:01 [INFO]   Trainable parameters: 8,388,608 (LoRA)
2025-10-27 14:30:01 [INFO]   Effective batch size: 32
2025-10-27 14:30:05 [INFO] Epoch 1/3, Step 100/300, Loss: 0.523, LR: 5.0e-5
...
```

---

## 🔬 **MPS Verification Script**

Create this script to diagnose MPS setup:

```bash
# Save as scripts/verify_mps.py
cat > scripts/verify_mps.py << 'PYTHON'
#!/usr/bin/env python3
"""
Verify MPS (Metal Performance Shaders) setup for Mac training.
"""

import sys
import torch

print("=" * 60)
print("MPS VERIFICATION")
print("=" * 60)

# 1. PyTorch version
print(f"\n1. PyTorch Version:")
print(f"   {torch.__version__}")

# 2. MPS availability
print(f"\n2. MPS Availability:")
print(f"   MPS built: {torch.backends.mps.is_built()}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("\n❌ MPS NOT AVAILABLE")
    print("   Possible fixes:")
    print("   - Upgrade PyTorch: pip install --upgrade torch")
    print("   - Check macOS version (requires macOS 12.3+)")
    print("   - Verify Apple Silicon Mac (not Intel)")
    sys.exit(1)

# 3. Test tensor operations
print(f"\n3. Testing MPS Operations:")
try:
    device = torch.device("mps")
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    print(f"   ✓ Matrix multiplication: {z.shape}")

    # Test gradient computation
    x.requires_grad = True
    loss = (x ** 2).sum()
    loss.backward()
    print(f"   ✓ Gradient computation: {x.grad.shape}")

    print(f"\n✅ MPS IS FULLY FUNCTIONAL!")
    print(f"\nYou can train with:")
    print(f"  python src/train.py --config <config.yaml> --device mps")
    print(f"  # or")
    print(f"  python src/train.py --config <config.yaml> --device auto")

except Exception as e:
    print(f"\n❌ MPS TEST FAILED: {e}")
    print(f"   Try using CPU fallback: --device cpu")
    sys.exit(1)

# 4. Memory info
print(f"\n4. Memory Info:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Total: {mem.total / (1024**3):.1f} GB")
    print(f"   Available: {mem.available / (1024**3):.1f} GB")
except ImportError:
    print("   (Install psutil for memory info: pip install psutil)")

print("\n" + "=" * 60)
PYTHON

chmod +x scripts/verify_mps.py
```

**Run it:**
```bash
python3 scripts/verify_mps.py
```

---

## 🎯 **Recommended Workflow for Your Mac Studio**

### **For Track A (Qwen 0.5B + Extended DPO):**

```bash
# Very comfortable - use recommended settings
python src/train.py \
  --config configs/v14/dpo_extended_100pairs.yaml \
  --device auto \
  --epochs 3

# Expected: 2-3 hours, ~12GB memory usage
```

### **For Track B (KTO Regularization):**

```bash
# Same as Track A, just different config
for lambda in 0.01 0.1 0.5; do
  python src/train.py \
    --config configs/v14/kto_regularized.yaml \
    --device auto \
    --kto-lambda $lambda \
    --output-dir checkpoints/v14/track_b/lambda_${lambda}
done

# Run overnight, compare results in morning
```

### **For Track C (Qwen 1.5B Scale-up):**

```bash
# Tighter memory - enable gradient checkpointing
python src/train.py \
  --config configs/v14/qwen25_1.5b.yaml \
  --device auto \
  --batch-size 2 \
  --gradient-accum 16 \
  --use-gradient-checkpointing

# Expected: 8-12 hours, ~28GB memory usage
```

---

## 💡 **Tips & Best Practices**

### **1. Monitor GPU Usage**

```bash
# Open Activity Monitor
open -a "Activity Monitor"

# In Activity Monitor:
# - Click "GPU" tab
# - Watch "GPU" column (should show high %)
# - Watch "Memory" column (should stay < 36GB)
```

### **2. Prevent Mac from Sleeping**

```bash
# Install caffeinate to keep Mac awake during training
caffeinate -i python src/train.py --config <config.yaml> --device mps

# Or use System Settings > Energy Saver > Prevent Mac from sleeping
```

### **3. Save Checkpoints Frequently**

```yaml
# In your config YAML
training:
  checkpoint_interval: 100  # Save every 100 steps
  save_total_limit: 3       # Keep only 3 latest checkpoints
```

### **4. Use WandB for Remote Monitoring**

```bash
# Install wandb
pip install wandb

# Login once
wandb login

# Then training logs will be accessible from any device:
# https://wandb.ai/your-username/phasegpt-v14
```

---

## 🆚 **Mac Studio vs. Cloud GPU Comparison**

| Factor | Mac Studio (M2 Ultra) | A100 (40GB) Cloud | Recommendation |
|--------|----------------------|-------------------|----------------|
| **Cost** | $0 (you own it) | ~$2-4/hour | Use Mac for experiments |
| **Speed** | 3-5x slower | Fast | Mac for <8 hour jobs |
| **Convenience** | Local, always on | Setup overhead | Mac for iterations |
| **Memory** | 36GB unified | 40GB VRAM | Similar capacity |
| **Best for** | Track A, B, quick tests | Track C, production | Hybrid approach |

**Recommended Strategy:**
1. **Develop on Mac:** Run Track A/B experiments locally (overnight jobs)
2. **Scale on Cloud:** If Track C (1.5B) is too slow, use cloud GPU for final run
3. **Evaluate on Mac:** All evaluation can run locally (inference is faster than training)

---

## 📚 **Resources**

- **PyTorch MPS Docs:** https://pytorch.org/docs/stable/notes/mps.html
- **Apple Metal Docs:** https://developer.apple.com/metal/pytorch/
- **PhaseGPT Configs:** `configs/v14/`
- **Training Script:** `src/train.py`

---

## ✅ **Checklist Before Training**

- [ ] PyTorch 2.1+ installed (`pip list | grep torch`)
- [ ] MPS available (`python3 scripts/verify_mps.py`)
- [ ] Dataset generated (`ls data/preferences_v14_100pairs.jsonl`)
- [ ] Config reviewed (`cat configs/v14/dpo_extended_100pairs.yaml`)
- [ ] Enough disk space (~10GB for checkpoints)
- [ ] Mac won't sleep during training (`caffeinate` or Energy Saver settings)
- [ ] Monitoring ready (Activity Monitor or WandB)

---

## 🚀 **Ready to Train!**

Your Mac Studio is **perfectly suited** for PhaseGPT Track A and B experiments. The 36GB unified memory gives you plenty of headroom for Qwen 0.5B and even allows Qwen 1.5B training.

**Start with:**
```bash
python3 scripts/verify_mps.py  # Verify MPS works
make validate-data              # Verify dataset
python src/train.py --config configs/v14/dpo_extended_100pairs.yaml --device auto
```

Good luck! 🍎🚀
