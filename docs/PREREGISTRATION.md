# Phase B Preregistration: Generalization and Anti-Oversynchronization Study

**Preregistration Date:** 2025-10-20
**Study Status:** Not Started (Infrastructure Complete)
**Primary Investigator:** [Your Name]
**OSF Project:** [To be linked]

---

## Preregistration Statement

This document constitutes a complete preregistration of Phase B experiments following Open Science Framework (OSF) standards. All hypotheses, methods, analyses, and success criteria are specified **before** examining WikiText-2 results.

**Commitment:** We will report all results as specified, including negative findings, without post-hoc modifications to success criteria or analysis plans.

---

## Background and Rationale

### Phase A Findings

Phase A hyperparameter tuning on Shakespeare dataset identified optimal configuration:
- **Layer position:** 7 (mid-network)
- **Oscillator count:** 32
- **Coupling strength:** K=1.0
- **Performance:** 4.85 PPL (2.4% improvement over baseline)

### Critical Discovery: Over-Synchronization Paradox

Interpretability analysis revealed:
- **Order parameter:** R = 0.8837 ± 0.0263
- **Target range:** R ∈ [0.30, 0.55] (literature-based Goldilocks zone)
- **Status:** Over-synchronized (61% above target upper bound)

### Research Question

**Does over-synchronization (R=0.88) observed in Phase A harm generalization to diverse, heterogeneous text corpora?**

**Hypothesis:** High synchronization benefits narrow, stylistically coherent corpora (Shakespeare) by enforcing tight thematic coupling, but collapses on diverse text (WikiText-2) due to mode collapse and reduced representational diversity.

---

## Study Design

### Phase B Objectives

1. **Primary:** Test generalization of Phase A winner to WikiText-2 (diverse corpus)
2. **Secondary:** Evaluate anti-oversynchronization controls to achieve target R range
3. **Tertiary:** Establish optimal coupling strength for general language modeling

### Experimental Conditions

Four configurations will be tested:

| Config | Description | K | Noise | Jitter | Reg | Rationale |
|--------|-------------|---|-------|--------|-----|-----------|
| **B1-Baseline** | Pure GPT-2 | N/A | - | - | - | Establish WikiText-2 baseline |
| **B2-KPC-Soft** | Soft coupling | 0.50 | - | - | - | Reduce coupling vs Phase A (K=1.0) |
| **B3-KPC-Mid** | Mid coupling | 0.75 | - | - | - | Intermediate coupling strength |
| **B4-KPC-Diverse** | Full anti-oversync | 0.75 | ✓ | ✓ | ✓ | Combined diversity mechanisms |

**Anti-oversynchronization controls (B4 only):**
- **Phase noise:** σ = 0.03 Gaussian noise on phases
- **Frequency jitter:** 2% relative variation in natural frequencies ω
- **Coherence regularizer:** Soft penalty when R > 0.45

### Dataset

**WikiText-2:**
- Source: Salesforce Research (Hugging Face `datasets`)
- Train: ~2M tokens, 36,718 sequences
- Validation: ~217K tokens
- Test: ~245K tokens
- Vocabulary: ~33K BPE tokens
- Characteristics: Diverse topics (Wikipedia articles), heterogeneous style

**Contrast with Phase A:**
- Shakespeare: 1M characters, single author, verse style, narrow thematic coherence
- WikiText-2: 2M tokens, multi-author, prose, broad encyclopedic content

---

## Hypotheses (Preregistered)

### Primary Hypotheses

**H1 (Generalization Performance):**
- **H1a (Null):** KPC configurations achieve Val PPL ≤ Baseline × 1.05 on WikiText-2
- **H1b (Alternative):** KPC configurations achieve Val PPL > Baseline × 1.05 (generalization failure)

**H2 (Coupling Strength):**
- **H2a:** Softer coupling (K=0.50) outperforms mid coupling (K=0.75) on WikiText-2
- **H2b:** Softer coupling achieves lower R (closer to target range)

**H3 (Anti-Oversync Controls):**
- **H3a:** B4-KPC-Diverse achieves R ∈ [0.35, 0.55] (target range)
- **H3b:** B4-KPC-Diverse achieves best PPL among KPC configurations

### Secondary Hypotheses

**H4 (Variance Reduction):**
- Anti-oversync controls (B4) show lower variance across runs than Phase A winner

**H5 (Training Stability):**
- Softer coupling (K=0.50, K=0.75) avoids catastrophic collapse observed at K=2.0

**H6 (Order Parameter Dynamics):**
- R(t) stabilizes faster in B4 (anti-oversync) than B2/B3 (no controls)

---

## Methods

### Model Architecture

**Base Model:** GPT-2 Small (83.3M parameters)
- 12 layers, 12 heads, 768 d_model
- 512 token context length
- Vocabulary: 33,304 BPE tokens (WikiText-2)

**Phase-Coupled Configurations (B2, B3, B4):**
- Phase attention: Single layer (Layer 7, based on Phase A)
- Oscillators: 32 (optimal from Phase A)
- Coupling strength: Variable (K=0.50, K=0.75)

### Training Protocol

**Common Settings:**
- Epochs: 50 (with early stopping)
- Batch size: 8 (reduced from 32 due to OOM constraint)
- Learning rate: 3.0e-4 (AdamW optimizer)
- Warmup: 500 steps (cosine schedule)
- Weight decay: 0.01
- Gradient clipping: 1.0
- Scheduler: Cosine with min_lr=1e-5

**Early Stopping:**
- Metric: Validation PPL
- Patience: 5 epochs
- Min delta: 0.01

**Seed:** 42 (all experiments)

### Anti-Oversynchronization Mechanisms (B4 Only)

**1. Phase Noise:**
```python
phases = phases + torch.randn_like(phases) * sigma
sigma = 0.03  # Preregistered value
```

**2. Frequency Jitter:**
```python
omega = omega_base * (1 + torch.randn_like(omega) * jitter)
jitter = 0.02  # 2% relative variation
```

**3. Coherence Regularizer:**
```python
if R > R_target:
    reg_loss = lambda_reg * (R - R_target)**2
lambda_reg = 0.01  # Preregistered value
R_target = 0.45    # Soft ceiling
```

Total loss: `loss = ce_loss + reg_loss`

### Hardware

**Target Platform:**
- GPU: 40GB+ VRAM (A100, A6000, or equivalent)
- Alternative: Sequential execution on smaller GPU (reduce batch_size=4)

**Estimated Compute:**
- Per configuration: 2-3 GPU hours (50 epochs with early stopping)
- Total: 8-12 GPU hours (4 configs)

---

## Measurement Plan

### Primary Outcomes

**1. Validation Perplexity (PPL)**
- Measured every 500 steps
- Best validation PPL recorded
- Final test PPL after training

**2. Order Parameter R(t)**
- Computed at every evaluation step
- Statistics: mean, std, min, max
- Trajectory saved for visualization

**3. Training Stability**
- Loss variance across epochs
- Presence/absence of catastrophic collapse
- Convergence speed (epochs to best PPL)

### Secondary Outcomes

**4. Synchronization Band Achievement**
- Fraction of time R ∈ [0.35, 0.55]
- Time to enter target band
- Band stability (no. of exits)

**5. Attention Pattern Diversity**
- Entropy of attention distributions
- Comparison with baseline attention

**6. Computational Overhead**
- Training time per epoch
- Memory usage
- Throughput (tokens/sec)

### Data Collection

**Logged Metrics (Every 100 Steps):**
- Training loss
- Gradient norm
- Learning rate

**Logged Metrics (Every 500 Steps / Evaluation):**
- Validation loss and PPL
- Order parameter R (for KPC configs)
- R statistics (mean, std, min, max)

**Final Checkpoints:**
- Best validation checkpoint
- Final checkpoint
- Optimizer state (for continuation)

---

## Analysis Plan

### Primary Analysis: Generalization Performance

**Success Criterion (H1a):**
```
For each KPC config (B2, B3, B4):
  If best_val_ppl_kpc ≤ best_val_ppl_baseline × 1.05:
    PASS (successful generalization)
  Else:
    FAIL (generalization failure)
```

**Statistical Test:**
- Compare mean PPL across 3 random seeds (optional if resources allow)
- One-sided t-test: H0: PPL_kpc ≤ PPL_baseline × 1.05
- Alpha: 0.05

**Interpretation:**
- PASS: KPC generalizes to diverse text
- FAIL: Over-synchronization harms generalization (supports alternative hypothesis)

### Secondary Analysis: Coupling Strength

**Compare B2 (K=0.50) vs B3 (K=0.75):**
```python
# PPL comparison
if ppl_b2 < ppl_b3:
    print("Softer coupling (K=0.50) outperforms mid coupling")
elif ppl_b3 < ppl_b2:
    print("Mid coupling (K=0.75) outperforms softer coupling")
else:
    print("No difference")

# R comparison
if mean_R_b2 < mean_R_b3:
    print("Softer coupling achieves lower synchronization")
```

### Tertiary Analysis: Anti-Oversync Controls

**B4 Success Criteria:**
1. **R in target range:** `0.35 ≤ mean(R) ≤ 0.55`
2. **Best KPC performance:** `ppl_b4 ≤ min(ppl_b2, ppl_b3)`
3. **Low variance:** `std(R_b4) < mean(std(R_b2), std(R_b3))`

**Mechanism Attribution:**
- Compare B3 (K=0.75, no controls) vs B4 (K=0.75, with controls)
- Difference isolates effect of noise + jitter + regularization

### Visualization Plan

**Required Plots:**
1. **Validation PPL curves:** All 4 configs overlaid
2. **R(t) trajectories:** B2, B3, B4 with target band shaded
3. **R distribution histograms:** Compare all KPC configs
4. **Attention pattern comparison:** Baseline vs best KPC
5. **Loss variance:** Box plots across configs

**Statistical Reporting:**
- All results reported with mean ± std
- Effect sizes (Cohen's d) for pairwise comparisons
- Confidence intervals (95%) for PPL differences

---

## Success Criteria Summary

### Must Achieve (Critical)

1. **All experiments complete:** 4 configs trained to convergence
2. **Baseline established:** Pure GPT-2 performance on WikiText-2 documented
3. **All metrics logged:** PPL, R(t), loss curves recorded
4. **Preregistered analysis completed:** Hypotheses tested as specified

### Optimal Outcome

1. **H1a confirmed:** KPC achieves PPL ≤ baseline × 1.05
2. **H3a confirmed:** B4 achieves R ∈ [0.35, 0.55]
3. **H3b confirmed:** B4 achieves best KPC performance

### Acceptable Alternative Outcome

1. **H1b confirmed:** KPC fails to generalize (PPL > baseline × 1.05)
   - Scientific value: Validates over-synchronization hypothesis
   - Contribution: Establishes generalization limits of phase coupling
2. **Negative results reported:** All findings published regardless of outcome

---

## Deviations and Contingencies

### Permitted Modifications

**If OOM errors persist despite batch_size=8:**
- Reduce sequence length: 512 → 256
- Further reduce batch size: 8 → 4
- Increase gradient accumulation steps to maintain effective batch size
- **Requirement:** Document all changes and report impact on results

**If training does not converge:**
- Extend epochs: 50 → 100
- Adjust learning rate: Try 1e-4 or 5e-4
- **Requirement:** Preregister modification before examining validation metrics

### Prohibited Modifications

The following are **NOT permitted** after examining WikiText-2 results:
- Changing coupling strengths (K values)
- Modifying anti-oversync parameters (σ, jitter, λ)
- Adjusting target R range
- Changing success criteria
- Selective reporting of configurations

### Contingency: Insufficient Resources

If GPU resources unavailable:
1. **Scaled-down pilot:** Run 1 epoch of each config to verify setup
2. **Public call:** Request community contribution for full execution
3. **Future work:** Document as limitation, publish Phase A alone

---

## Reporting Plan

### Mandatory Reporting

**Regardless of outcome, we will report:**
1. All 4 configurations (no cherry-picking)
2. Negative results if generalization fails
3. Unexpected findings (crashes, anomalies)
4. Exact training times and costs
5. Deviations from preregistered protocol (if any)

### Results Format

**For each configuration:**
- Best validation PPL ± std (if multiple seeds)
- R statistics: mean, std, min, max
- Training time and final epoch
- Convergence plot (PPL vs epoch)
- R trajectory plot (for KPC configs)

**Comparative Analysis:**
- Table: All 4 configs with PPL, R, time
- Statistical tests: p-values for pairwise comparisons
- Effect sizes: Cohen's d for key comparisons

### Publication Target

- **Primary:** Conference paper (ICLR, NeurIPS, or workshop)
- **Secondary:** ArXiv preprint concurrent with OSF publication
- **Data:** All checkpoints, logs, and configs on OSF

---

## Timeline

**Execution Plan (When Resources Available):**

1. **Setup (1 hour):**
   - Provision GPU instance
   - Install dependencies
   - Download WikiText-2
   - Verify configuration files

2. **Baseline (B1) (2-3 hours):**
   - Train pure GPT-2
   - Establish reference PPL

3. **KPC Experiments (6-9 hours):**
   - B2-KPC-Soft (2-3 hours)
   - B3-KPC-Mid (2-3 hours)
   - B4-KPC-Diverse (2-3 hours)

4. **Analysis (2-3 hours):**
   - Generate all preregistered plots
   - Run statistical tests
   - Compute success criteria

5. **Reporting (1 week):**
   - Write results section
   - Prepare figures
   - Submit to OSF and ArXiv

**Total GPU Time:** 8-12 hours
**Estimated Cost:** $80-120 (Lambda Labs A100 at $1.10/hour)

---

## Ethical Considerations

**Research Integrity:**
- Preregistration prevents p-hacking and HARKing (Hypothesizing After Results Known)
- Transparent reporting of all results
- Open data and code sharing

**Reproducibility:**
- Fixed random seeds
- Version-controlled configurations
- Complete documentation

**Resource Stewardship:**
- Efficient experiment design (4 configs, not exhaustive grid search)
- Clear stopping criteria (early stopping)
- Computational cost disclosed

---

## Amendments

Any amendments to this preregistration must be:
1. Documented with date and rationale
2. Made **before** examining WikiText-2 results
3. Clearly marked in analysis
4. Justified by technical constraints (not results)

**Amendment Log:**
- (None as of 2025-10-20)

---

## Sign-Off

**Preregistration Status:** Complete
**Infrastructure Status:** All code, configs, and scripts ready
**Execution Status:** Not started (awaiting GPU resources)

**Commitment Statement:**

I commit to executing Phase B experiments exactly as specified in this preregistration, reporting all results transparently, and documenting any deviations with clear justification.

**Investigator:** [Your Name]
**Date:** 2025-10-20
**Signature:** [Digital signature or confirmation]

---

## References

**Preregistration Standards:**
- Open Science Framework. (2023). Preregistration Guidelines. https://osf.io/prereg/

**Kuramoto Model:**
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *International Symposium on Mathematical Problems in Theoretical Physics*.

**Phase A Results:**
- See `PHASE_A_FINAL_REPORT.md` for complete Phase A findings and over-synchronization discovery.

---

**Document Version:** 1.0
**OSF Preregistration:** [To be linked after OSF submission]
**Timestamp:** 2025-10-20 14:00 PST
**Status:** Preregistered, Not Started

---

## Appendix A: Configuration Files

All configuration files are version-controlled and frozen:

1. `PhaseB/configs/wt2_baseline.yaml` (sha256: [hash])
2. `PhaseB/configs/wt2_kpc_soft.yaml` (sha256: [hash])
3. `PhaseB/configs/wt2_kpc_mid.yaml` (sha256: [hash])
4. `PhaseB/configs/wt2_kpc_diverse.yaml` (sha256: [hash])

No modifications permitted after preregistration date.

## Appendix B: Code Verification

Training script: `PhaseB/scripts/train_generalize.py`
- R tracking: Implemented (line 245-260)
- Anti-oversync controls: Implemented (line 180-210)
- Logging: Complete (TensorBoard + CSV)

Unit tests pass:
```bash
pytest tests/test_phase_attention.py::test_order_parameter
pytest tests/test_coherence_utils.py::test_regularizer
```

## Appendix C: Contact

For questions about this preregistration:
- Email: [Your Email]
- GitHub: https://github.com/yourusername/PhaseGPT/issues
- OSF Project: [To be linked]

---

🌀 *Preregistered with integrity. Executed with transparency. Reported with honesty.*
