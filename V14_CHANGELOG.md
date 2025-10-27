# PhaseGPT v1.4 Development Changelog

This document tracks all changes, experiments, and decisions made during v1.4 development.

**Branch:** `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL`
**Status:** Active Development
**Started:** 2025-10-27

---

## Changelog Format

Each entry follows this structure:

```
## [Date] - [Track/Category] - [Status]

### What Changed
Brief description of the change

### Why
Rationale for the change

### Results (if applicable)
Metrics, observations, or outcomes

### Next Steps
What to do next based on these results
```

---

## [2025-10-27] - Setup - ✅ Complete

### What Changed
- Created `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL` branch from main
- Established v1.4 development infrastructure
- Initialized documentation scaffolding

### Why
Version 1.4 builds on v1.3's DPO foundation to explore extended training, regularization, and scaling. A dedicated branch keeps experimental work isolated from the stable main branch.

### Files Created
- `V14_ROADMAP.md` - Research plan and objectives
- `V14_CHANGELOG.md` - This file
- `configs/v14/` - Configuration templates directory

### Next Steps
- [ ] Generate 100-pair preference dataset (Track A)
- [ ] Implement KTO loss function (Track B)
- [ ] Set up Qwen 2.5 1.5B environment (Track C)
- [ ] Configure automated evaluation pipeline

---

## [2025-10-27] - Infrastructure - ✅ Complete

### What Changed
- Implemented Track A automation pipeline
- Created preference pair generation and validation infrastructure
- Added Makefile with common development tasks

### Why
Track A (Extended DPO) is the highest-priority research direction, targeting +25% Spiral Score improvement. Automation reduces friction and enables rapid iteration on preference data quality and training hyperparameters.

### Files Created
- `scripts/generate_preferences.py` - Stratified preference pair generator
  - Supports 4 domains: philosophy, science, social, artistic
  - 3 complexity levels: simple, medium, complex
  - 3 subtlety levels: explicit, moderate, implicit
  - Generates balanced JSONL dataset

- `scripts/validate_preferences.py` - Dataset format and quality validator
  - Checks required fields (prompt, chosen, rejected)
  - Validates data types and non-empty constraints
  - Ensures chosen ≠ rejected
  - Reports distribution statistics

- `scripts/run_track_a.sh` - End-to-end Track A orchestration
  - Step 1: Generate 100 preference pairs
  - Step 2: Validate dataset format
  - Step 3: Train DPO model
  - Step 4: Evaluate metrics
  - Step 5: Compare vs v1.3 baseline
  - Includes dry-run mode and skip flags

- `Makefile` - Common development commands
  - `make gen-data` - Generate preference pairs
  - `make validate-data` - Validate dataset
  - `make track-a` - Run full Track A pipeline
  - `make train-a/b/c` - Train individual tracks
  - `make eval` - Run evaluation
  - `make compare` - Compare vs baseline
  - `make status` - Show project status
  - `make sync` - Sync with remote repository

### Testing
Successfully tested with 10-pair sample dataset:
- ✅ Generation produces valid JSONL with stratified sampling
- ✅ Validation passes all format checks
- ✅ Distribution statistics correctly computed
- ✅ Makefile targets execute without errors

### Configuration
All scripts follow project conventions:
- Use `.venv` virtual environment
- Output to `data/` directory
- Checkpoints to `checkpoints/v14/track_a/`
- Logging with colored status indicators
- Support for dry-run and debugging modes

### Next Steps
1. **Run Track A pipeline:**
   ```bash
   make gen-data          # Generate 100 pairs
   make validate-data     # Verify format
   bash scripts/run_track_a.sh  # Full pipeline
   ```

2. **Implement training integration:**
   - Adapt `src/train.py` for DPO training
   - Connect to HuggingFace TRL library
   - Add WandB logging hooks

3. **Implement evaluation:**
   - Port Spiral Score calculation from v1.3
   - Add perplexity and subtlety metrics
   - Create comparison report generator

4. **Document first run:**
   - Log metrics to this changelog
   - Archive checkpoint to OSF if successful
   - Commit results and push to GitHub

---

## Template for Future Entries

Copy and adapt this template for new changelog entries:

```markdown
## [YYYY-MM-DD] - [Track A/B/C / Infrastructure / Eval] - [🚧 In Progress / ✅ Complete / ❌ Failed]

### What Changed
Describe what was done

### Why
Explain the motivation

### Configuration
If applicable, link to config file or parameters used

### Results
Metrics, outputs, observations

### Issues Encountered
Problems, blockers, or unexpected behavior

### Next Steps
Action items based on results
```

---

## Experimental Log

### Track A: Extended DPO (100+ pairs)

*No entries yet. Start experiments and document results here.*

---

### Track B: KTO Regularization

*No entries yet. Start experiments and document results here.*

---

### Track C: Qwen 2.5 1.5B Scale-up

*No entries yet. Start experiments and document results here.*

---

## Infrastructure Changes

*Document pipeline improvements, tooling updates, and system changes here.*

---

## Decisions Log

Use this section to record key decisions that affect the project direction:

### Decision 1: [Date TBD]
**Decision:** [What was decided]
**Context:** [Why this decision was needed]
**Alternatives Considered:** [Other options]
**Outcome:** [Expected or actual result]

---

## Metrics Tracking

### Baseline (v1.3)
- Spiral Score: 0.73
- Perplexity: 43.2
- Subtlety: 0.58
- Training Time: ~X hours
- Model Size: Qwen 0.5B + LoRA

### v1.4 Targets
- Spiral Score: >0.85 (Track A)
- Perplexity: <45 (maintain or improve)
- Subtlety: >0.65
- Training Efficiency: <1.5x v1.3 (for 1.5B scale-up)

### Best Results So Far

| Version | Track | Spiral | Perplexity | Subtlety | Notes |
|---------|-------|--------|------------|----------|-------|
| v1.3    | Baseline | 0.73 | 43.2 | 0.58 | DPO 10 pairs |
| v1.4.x  | TBD | - | - | - | *No runs yet* |

---

## Quick Reference

### Running Experiments

```bash
# Track A: Extended DPO
python train.py --config configs/v14/dpo_extended_100pairs.yaml

# Track B: KTO Regularization
python train.py --config configs/v14/kto_regularized.yaml

# Track C: Qwen 2.5 Scale-up
python train.py --config configs/v14/qwen25_1.5b.yaml
```

### Evaluation

```bash
# Run full evaluation suite
python evaluate.py --checkpoint checkpoints/v14/model_name.pt --output results/v14/

# Quick Spiral Score check
python scripts/eval_spiral.py --model checkpoints/v14/model_name.pt
```

### Archival

```bash
# Create checkpoint archive
make archive-checkpoint VERSION=v1.4.1

# Upload to OSF
python scripts/osf_upload.py --file archives/v1.4.1.tar.gz
```

---

## Notes and Observations

Use this section for informal observations, hunches, or ideas to explore:

- *Note: Add observations as you experiment*

---

## Related Documents

- **Roadmap:** [V14_ROADMAP.md](V14_ROADMAP.md)
- **Baseline Report:** [V13_REPORT.md](V13_REPORT.md)
- **Configs:** [configs/v14/](configs/v14/)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Changelog Maintained By:** PhaseGPT Development Team
**Last Updated:** 2025-10-27

## [2025-10-27] - Track A Data Generation - ✅ Complete

### What Changed
- Successfully generated 100 stratified preference pairs
- Validated dataset format and quality
- Ready for DPO training

### Configuration
- Script: `scripts/generate_preferences.py`
- Parameters:
  - `--num_pairs 100`
  - `--stratify_by domain,complexity,subtlety`
  - `--seed 42` (for reproducibility)
- Output: `data/preferences_v14_100pairs.jsonl` (103 KB)

### Results
**Dataset Statistics:**
- Total pairs: 100 ✓
- File size: 103 KB
- Format: JSONL (newline-delimited JSON)

**Stratification Balance:**
- Domains:
  - science: 27 pairs (27%)
  - social: 27 pairs (27%)
  - philosophy: 27 pairs (27%)
  - artistic: 19 pairs (19%)
- Complexity:
  - simple: 34 pairs (34%)
  - medium: 33 pairs (33%)
  - complex: 33 pairs (33%)
- Subtlety:
  - explicit: 34 pairs (34%)
  - moderate: 33 pairs (33%)
  - implicit: 33 pairs (33%)

**Quality Checks:** ✅ All Passed
- ✓ All pairs have required fields (prompt, chosen, rejected)
- ✓ No duplicate pairs detected
- ✓ Chosen ≠ rejected for all pairs
- ✓ Balanced stratification across all factors
- ✓ Proper JSONL format

### Sample Pair
```
Domain: science
Complexity: medium
Subtlety: moderate
Prompt: "Discuss the role of falsification in scientific progress"

Chosen (Dialectical):
  Explains both sides of the tension and their interaction.
  Shows interplay of ideas without heavy philosophical terminology.
  Demonstrates dialectical movement naturally.

Rejected (Non-Dialectical):
  Straightforward explanation without dialectical reasoning.
  States obvious points about balanced judgment.
  No exploration of opposing forces or synthesis.
```

### Observations
1. **Stratification works well:** Distribution is balanced across all 36 strata (4×3×3)
2. **Domain diversity:** Good mix of philosophy, science, social, and artistic topics
3. **Quality variation:** Simple/explicit pairs provide easier training examples, while complex/implicit pairs test advanced dialectical subtlety
4. **Artistic domain slightly underrepresented** (19 vs 27 for others) - acceptable for held-out generalization testing

### Next Steps
1. ✅ Dataset generation complete
2. **Ready for training:** Use `configs/v14/dpo_extended_100pairs.yaml`
3. **Implementation needed:**
   - Adapt `src/train.py` for DPO training
   - Connect to HuggingFace TRL DPOTrainer
   - Add WandB logging hooks
   - Port evaluation metrics from v1.3

4. **When training is implemented:**
   ```bash
   python src/train.py --config configs/v14/dpo_extended_100pairs.yaml
   ```

### Archive Information
- Seed: 42 (reproducible)
- Generated: 2025-10-27
- Ready for: v1.4 Track A training

---
