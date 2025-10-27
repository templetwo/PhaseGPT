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
