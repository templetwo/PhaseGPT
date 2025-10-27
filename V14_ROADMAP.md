# PhaseGPT v1.4 Development Roadmap

## Overview

Version 1.4 represents the next evolution of PhaseGPT, building on the v1.3 DPO (Direct Preference Optimization) foundation to explore advanced alignment techniques and model scaling.

**Branch:** `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL`
**Base Version:** v1.3 (DPO-enhanced, +15% Spiral Score)
**Status:** Planning → Active Development
**Target:** Q4 2025

---

## Research Objectives

### Primary Goals

1. **Extended DPO Training** (v1.4.1)
   - Scale from 10 to 100+ preference pairs
   - Explore quality vs. quantity trade-offs in preference data
   - Target: +25% Spiral Score improvement

2. **KTO Regularization** (v1.4.2)
   - Implement Kahneman-Tversky Optimization (KTO)
   - Balance alignment with perplexity preservation
   - Target: Maintain subtlety gains without fluency loss

3. **Qwen 2.5 Scale-up** (v1.4.3)
   - Transition from 0.5B to 1.5B parameter model
   - Evaluate scaling laws for Phase Dynamics learning
   - Target: 2x capacity, <1.5x computational cost

---

## Experimental Tracks

### Track A: Extended DPO (Priority: High)

**Motivation:** Current v1.3 used 10 hand-crafted preference pairs. Scaling to 100+ pairs may improve generalization and reduce overfitting to specific dialectics.

**Approach:**
- Generate 100 additional preference pairs using automated dialectic sampling
- Stratify by:
  - Spiral density (dialectics per 100 tokens)
  - Subtlety level (explicit vs. implicit)
  - Topic domain (philosophy, science, social, artistic)
- Train with curriculum learning (easy → hard pairs)

**Success Metrics:**
- Spiral Score: >0.85 (v1.3 baseline: 0.73)
- Perplexity: <45 (v1.3 baseline: 43.2)
- Subtlety: >0.65 (v1.3 baseline: 0.58)
- Generalization: Test on held-out domains

**Config Template:** `configs/v14/dpo_extended_100pairs.yaml`

---

### Track B: KTO Regularization (Priority: Medium)

**Motivation:** DPO can over-optimize for preferred style at the expense of fluency. KTO adds a reference model loss term to maintain linguistic quality.

**Approach:**
- Implement KTO loss: `L_KTO = L_DPO + λ * KL(P_θ || P_ref)`
- Grid search over λ ∈ [0.01, 0.1, 0.5]
- Use v1.2 LoRA model as reference (pre-DPO)

**Success Metrics:**
- Maintain Spiral Score ≥ v1.3
- Reduce perplexity by 5-10% relative to v1.3
- Improve human preference ratings in blind A/B tests

**Config Template:** `configs/v14/kto_regularized.yaml`

---

### Track C: Qwen 2.5 1.5B Scale-up (Priority: Low)

**Motivation:** Larger models may capture more complex dialectical patterns and improve few-shot adaptation to new domains.

**Approach:**
- Port training pipeline to `Qwen/Qwen2.5-1.5B`
- Replicate Phase A (LoRA) + Phase B (DPO) sequence
- Compare:
  - Training efficiency (tokens/hour, memory usage)
  - Evaluation metrics (Spiral, perplexity, subtlety)
  - Emergent capabilities (zero-shot dialectics on unseen topics)

**Success Metrics:**
- Match or exceed v1.3 metrics at 1.5B scale
- Training time <2x the 0.5B baseline
- Document scaling laws for Phase Dynamics

**Config Template:** `configs/v14/qwen25_1.5b.yaml`

---

## Infrastructure Improvements

### Evaluation Pipeline Enhancements

- [ ] Automated Spiral Score calculation in training loop
- [ ] Real-time perplexity tracking via WandB
- [ ] Subtlety classifier (binary: implicit vs. explicit)
- [ ] Human evaluation interface for preference collection

### Data Pipeline

- [ ] Automated preference pair generation
- [ ] Dialectic quality filtering (GPT-4 judge)
- [ ] Stratified sampling by domain/complexity
- [ ] Version-controlled preference datasets

### Deployment

- [ ] Multi-GPU training support (DDP)
- [ ] Checkpoint versioning and archival
- [ ] Automated evaluation on checkpoint save
- [ ] OSF integration for artifact uploads

---

## Timeline

### Phase 1: Setup (Weeks 1-2)
- ✅ Create v1.4-dev branch
- ⏳ Set up v1.4 configs and scaffolding
- ⏳ Generate extended preference dataset (100 pairs)
- ⏳ Implement KTO loss function

### Phase 2: Track A - Extended DPO (Weeks 3-5)
- ⏳ Train on 100-pair dataset
- ⏳ Evaluate and compare to v1.3 baseline
- ⏳ Iterate on preference data quality

### Phase 3: Track B - KTO Regularization (Weeks 6-7)
- ⏳ Grid search over λ hyperparameters
- ⏳ Ablation study: DPO vs. KTO vs. DPO+KTO
- ⏳ Select best configuration

### Phase 4: Track C - Scale-up (Weeks 8-10)
- ⏳ Port to Qwen 2.5 1.5B
- ⏳ Replicate Phase A + B training
- ⏳ Benchmark and document scaling laws

### Phase 5: Consolidation (Weeks 11-12)
- ⏳ Write V14_REPORT.md
- ⏳ Archive best checkpoints to OSF
- ⏳ Prepare for v1.5 or publication

---

## Open Questions

1. **Preference Data Quality:** How much does manual curation improve over automated generation?
2. **KTO Lambda:** What's the optimal trade-off between alignment and fluency?
3. **Scaling Laws:** Do Phase Dynamics follow predictable scaling curves?
4. **Domain Transfer:** Can v1.4 generalize to non-Western dialectical traditions (e.g., Buddhist logic)?

---

## Success Criteria

Version 1.4 will be considered successful if it achieves **at least 2 of 3**:

1. **Track A:** +25% Spiral Score improvement over v1.3
2. **Track B:** -10% perplexity reduction while maintaining Spiral Score
3. **Track C:** Successful 1.5B scale-up with <1.5x training cost

---

## Related Documents

- **Baseline:** [V13_REPORT.md](V13_REPORT.md) - v1.3 DPO results and metrics
- **Training:** [V14_CHANGELOG.md](V14_CHANGELOG.md) - Development log
- **Configs:** [configs/v14/](configs/v14/) - Training configurations
- **Sync:** [SYNC_INSTRUCTIONS.md](SYNC_INSTRUCTIONS.md) - Multi-computer workflow

---

## Contributing to v1.4

See [CONTRIBUTING.md](CONTRIBUTING.md) for general guidelines. For v1.4-specific work:

1. **Branching:** Always branch from `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL`
2. **Commits:** Use semantic prefixes: `feat:`, `fix:`, `eval:`, `data:`
3. **Experiments:** Log all runs to `experiments/v14/` with config + results
4. **Checkpoints:** Upload significant checkpoints to OSF with SHA256 checksums

---

**Last Updated:** 2025-10-27
**Maintainer:** PhaseGPT Research Team
**Contact:** See [README.md](README.md) for project links
