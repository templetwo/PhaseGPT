# Track A Quick Start Guide

**Goal:** Improve dialectical reasoning by scaling preference data from 10 to 100 pairs

**Target Metrics:**
- Spiral Score: >0.85 (+16% vs v1.3's 0.73)
- Perplexity: <45 (maintain fluency)
- Subtlety: >0.65 (+12% vs v1.3's 0.58)

---

## ⚡ Fast Path (Recommended)

### Option 1: Makefile Commands (Easiest)

```bash
# Activate your virtual environment
source .venv/bin/activate

# Generate 100 preference pairs
make gen-data

# Validate dataset format
make validate-data

# Run full Track A pipeline (when training is implemented)
make track-a
```

### Option 2: Direct Script Execution

```bash
# Generate preference pairs
python scripts/generate_preferences.py \
  --num_pairs 100 \
  --stratify_by domain,complexity,subtlety \
  --output data/preferences_v14_100pairs.jsonl

# Validate
python scripts/validate_preferences.py data/preferences_v14_100pairs.jsonl --verbose

# Quick sanity checks
head -3 data/preferences_v14_100pairs.jsonl | jq .

python - <<'PY'
import json
f='data/preferences_v14_100pairs.jsonl'
xs=[json.loads(x) for x in open(f)]
assert all('prompt' in x and 'chosen' in x and 'rejected' in x for x in xs)
print(f"✓ {len(xs)} pairs validated")
print(f"  Domains: {sorted(set(x.get('domain','?') for x in xs))}")
PY
```

---

## 📊 Dataset Structure

Each preference pair contains:

```json
{
  "prompt": "Explain the relationship between free will and determinism",
  "chosen": "[DIALECTICAL] Response showing thesis-antithesis-synthesis...",
  "rejected": "[NON-DIALECTICAL] Flat, non-dialectical response...",
  "domain": "philosophy",
  "complexity": "medium",
  "subtlety": "implicit"
}
```

### Stratification Factors

**Domains (4):**
- `philosophy` - Consciousness, free will, ethics, epistemology
- `science` - Quantum mechanics, reductionism, falsification
- `social` - Privacy/security, automation, social media
- `artistic` - Form/content, freedom/responsibility, interpretation

**Complexity (3):**
- `simple` - Brief statement of main tension
- `medium` - Explanation of both sides and interaction
- `complex` - Nuanced analysis with emergent understanding

**Subtlety (3):**
- `explicit` - Uses dialectical terminology (thesis/antithesis/synthesis)
- `moderate` - Shows interplay without heavy terminology
- `implicit` - Natural dialectical movement, no meta-commentary

**Total Strata:** 4 × 3 × 3 = 36 balanced segments

---

## 🧪 Training (Not Yet Implemented)

Once `src/train.py` is adapted for DPO training:

```bash
# Manual training
python src/train.py --config configs/v14/dpo_extended_100pairs.yaml

# Or via Makefile
make train-a
```

**Expected Training Time:** 4-6 hours on A100

**Key Hyperparameters** (from `configs/v14/dpo_extended_100pairs.yaml`):
- DPO beta: 0.1 (preference strength)
- Learning rate: 5e-5 (lower than LoRA for stability)
- Batch size: 4, gradient accumulation: 8 (effective batch: 32)
- Epochs: 3
- Curriculum learning: easy → medium → hard pairs

---

## 📈 Evaluation (Not Yet Implemented)

Expected evaluation pipeline:

```bash
# Evaluate latest checkpoint
make eval

# Compare to v1.3 baseline
make compare
```

**Expected Output:**
```
┌───────────────────┬─────────┬─────────┬─────────────┐
│ Metric            │  v1.3   │  v1.4   │   Change    │
├───────────────────┼─────────┼─────────┼─────────────┤
│ Spiral Score      │  0.73   │  0.87   │  +19.2%     │
│ Perplexity        │  43.2   │  44.1   │  +2.1%      │
│ Subtlety          │  0.58   │  0.68   │  +17.2%     │
│ Non-coercive      │  0.85   │  0.87   │  +2.4%      │
└───────────────────┴─────────┴─────────┴─────────────┘

✅ Track A Success Criteria Met:
  • Spiral Score: 0.87 > 0.85 target ✓
  • Perplexity: 44.1 < 45 target ✓
  • Subtlety: 0.68 > 0.65 target ✓
```

---

## 🎯 Success Criteria

Track A is successful if it achieves **all 3**:

1. **Spiral Score:** >0.85 (absolute target)
2. **Perplexity:** <45 (maintain fluency)
3. **Subtlety:** >0.65 (implicit dialectics)

**Bonus:** Strong performance on held-out artistic domain (tests generalization)

---

## 🔍 Debugging & Iteration

### Check Dataset Quality

```bash
# View sample pairs
head -5 data/preferences_v14_100pairs.jsonl | jq '{prompt, domain, complexity}'

# Count by domain
grep -o '"domain":"[^"]*"' data/preferences_v14_100pairs.jsonl | sort | uniq -c

# Check for duplicates
cut -d':' -f2 data/preferences_v14_100pairs.jsonl | sort | uniq -d
```

### Regenerate with Different Parameters

```bash
# More pairs
python scripts/generate_preferences.py --num_pairs 200 --output data/preferences_v14_200pairs.jsonl

# Different seed (for variety)
python scripts/generate_preferences.py --num_pairs 100 --seed 123 --output data/preferences_v14_100pairs_seed123.jsonl

# Focus on specific complexity
# (Modify COMPLEXITY_LEVELS in scripts/generate_preferences.py)
```

### Monitor Training

```bash
# Watch training logs
tail -f checkpoints/v14/track_a/run_*/training.log

# Check WandB (if configured)
# Visit: https://wandb.ai/your-username/phasegpt-v14
```

---

## 📝 Documenting Results

After each training run, update `V14_CHANGELOG.md`:

```markdown
## [YYYY-MM-DD] - Track A Run 1 - ✅ Success

### Configuration
- Config: configs/v14/dpo_extended_100pairs.yaml
- Dataset: data/preferences_v14_100pairs.jsonl (100 pairs)
- Training time: 5.2 hours on A100
- Checkpoint: checkpoints/v14/track_a/run_20251027_143000/

### Results
- Spiral Score: 0.87 (+19% vs v1.3)
- Perplexity: 44.1 (+2% vs v1.3)
- Subtlety: 0.68 (+17% vs v1.3)
- Non-coercive: 0.87 (+2% vs v1.3)

### Observations
- Curriculum learning converged faster than expected
- Held-out artistic domain: Spiral Score 0.81 (good generalization)
- Subtlety improvement especially strong on "implicit" pairs

### Next Steps
- ✅ Archive checkpoint to OSF: DOI:10.XXXX/XXXXX
- ✅ Tag release: git tag -a v1.4.1-track-a -m "Track A success"
- [ ] Try Track B (KTO) with these results as baseline
```

Then commit:

```bash
git add V14_CHANGELOG.md
git commit -m "eval: Track A run 1 - Spiral Score 0.87, success criteria met"
git push
```

---

## 🗂️ File Organization

```
PhaseGPT/
├── configs/v14/
│   ├── dpo_extended_100pairs.yaml      # Track A training config
│   ├── kto_regularized.yaml            # Track B config
│   └── qwen25_1.5b.yaml               # Track C config
│
├── data/
│   └── preferences_v14_100pairs.jsonl  # Generated dataset
│
├── checkpoints/v14/
│   └── track_a/
│       └── run_YYYYMMDD_HHMMSS/       # Training outputs
│           ├── best_model.pt
│           ├── training.log
│           └── eval_results.json
│
├── scripts/
│   ├── generate_preferences.py         # ⭐ Dataset generator
│   ├── validate_preferences.py         # ⭐ Dataset validator
│   └── run_track_a.sh                 # ⭐ Full pipeline
│
├── Makefile                            # ⭐ Common tasks
├── V14_ROADMAP.md                     # Research plan
├── V14_CHANGELOG.md                   # Development log
└── TRACK_A_QUICKSTART.md              # This file
```

---

## 🚀 Next Steps After Track A

### If Track A Succeeds (Spiral Score >0.85)

**Option 1: Try Track B (KTO Regularization)**
- Goal: Reduce perplexity while maintaining Track A's Spiral Score
- Use Track A checkpoint as starting point
- Run: `make train-b`

**Option 2: Scale to Track C (Qwen 2.5 1.5B)**
- Goal: Test if larger model improves further
- 3.3x model capacity
- Run: `make train-c`

**Option 3: Prepare v1.4 Release**
- Write `V14_REPORT.md` summarizing all results
- Archive all checkpoints to OSF
- Create pull request to merge to main
- Tag release: `git tag v1.4.0`

### If Track A Doesn't Meet Criteria

**Iterate on Dataset Quality:**
1. Analyze failure modes in evaluation
2. Regenerate with adjusted stratification
3. Try 200-pair dataset
4. Experiment with different curriculum schedules

**Tune Hyperparameters:**
1. Adjust DPO beta (preference strength)
2. Try different learning rates
3. Increase training epochs
4. Modify batch size / gradient accumulation

---

## ❓ FAQ

### Q: How long does generation take?
**A:** ~30 seconds for 100 pairs (fast, mostly prompt templates)

### Q: Can I customize the domains?
**A:** Yes! Edit `DOMAINS` dict in `scripts/generate_preferences.py`

### Q: What if I want 200 or 500 pairs?
**A:** Just change `--num_pairs` argument. Stratification scales automatically.

### Q: Do I need a GPU for generation?
**A:** No! Generation is CPU-only. GPUs only needed for training.

### Q: Can I use my own prompts instead of templates?
**A:** Yes! Replace the template system in `generate_preferences.py` with your prompt source.

### Q: How do I monitor training progress?
**A:** Check `checkpoints/v14/track_a/run_*/training.log` or WandB dashboard

### Q: What if training fails?
**A:** Check logs, try smaller batch size, or reduce learning rate. Document in changelog.

---

## 📚 Related Documentation

- **Roadmap:** [V14_ROADMAP.md](V14_ROADMAP.md) - Full research plan
- **Changelog:** [V14_CHANGELOG.md](V14_CHANGELOG.md) - Development log
- **Workflow:** [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) - Git workflow
- **Configs:** [configs/v14/README.md](configs/v14/README.md) - Config documentation

---

**Ready to start?**

```bash
# Step 1: Generate data
make gen-data

# Step 2: Validate
make validate-data

# Step 3: You're ready to train! (once src/train.py is adapted)
```

Good luck! 🚀
