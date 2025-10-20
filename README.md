# PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A systematic investigation of Kuramoto phase-coupled oscillator mechanisms in transformer attention layers for language modeling. This project presents the first rigorous hyperparameter study demonstrating that phase synchronization dynamics can improve language model performance while revealing critical insights about over-synchronization risks.

## Key Findings

**Phase A Results (Complete):**
- **2.4% improvement** in perplexity (4.85 vs 4.97 baseline) on Shakespeare dataset
- **Optimal configuration:** Single layer (layer 7), 32 oscillators, K=1.0 coupling strength
- **Goldilocks principle:** 32 oscillators optimal (16 unstable, 64 catastrophic)
- **Over-synchronization paradox:** High coherence (R=0.88) achieved strong performance on narrow corpus but raises generalization concerns

**Critical Discovery:** Strong coupling (K=2.0) causes catastrophic training collapse, dropping from 4.94 PPL to 9.21 PPL after epoch 20.

## Project Status

- **Phase A (Hyperparameter Tuning):** ✅ Complete - 7 configurations systematically tested
- **Phase B (Generalization Testing):** 🔄 Infrastructure ready, experiments not executed due to resource constraints

## What is Kuramoto Phase-Coupled Attention?

The Kuramoto model describes how coupled oscillators spontaneously synchronize, explaining phenomena from firefly flashing to neuronal activity. This project integrates Kuramoto dynamics into transformer attention:

```
Each attention head maintains N oscillators with phases θ_i(t)
Synchronization emerges through coupling: dθ_i/dt = ω_i + (K/N)Σ sin(θ_j - θ_i)
Attention weights modulated by phase coherence
```

**Hypothesis:** Phase synchronization provides an inductive bias for capturing long-range dependencies and semantic coherence in language.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate phasegpt
```

### Reproduce Phase A Winner

```bash
# Train Phase A optimal configuration (Layer 7, 32 oscillators, K=1.0)
python src/train.py \
    --config configs/phase_a_winner.yaml \
    --device cuda \
    --epochs 20

# Evaluate on test set
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --device cuda
```

### Analyze Synchronization Dynamics

```bash
# Run interpretability analysis
python PhaseB/scripts/interpret_model.py \
    --checkpoint checkpoints/best_model.pt \
    --num_tokens 512 \
    --output_dir results/interpretability/
```

## Repository Structure

```
PhaseGPT/
├── src/                          # Core implementation
│   ├── model.py                  # GPT-2 base model with phase attention support
│   ├── phase_attention.py        # Kuramoto phase-coupled attention mechanism
│   ├── coherence_utils.py        # Order parameter tracking and regularization
│   ├── train.py                  # Training loop with R(t) logging
│   ├── evaluate.py               # Evaluation and perplexity calculation
│   └── data.py                   # Dataset utilities (Shakespeare, WikiText-2)
│
├── configs/                      # Experiment configurations
│   ├── phase_a_winner.yaml       # Optimal Phase A configuration
│   ├── baseline.yaml             # Standard GPT-2 baseline
│   └── phase_b/                  # Phase B generalization configs (not run)
│       ├── wt2_baseline.yaml
│       ├── wt2_kpc_soft.yaml     # K=0.50 softer coupling
│       ├── wt2_kpc_mid.yaml      # K=0.75 mid-range
│       └── wt2_kpc_diverse.yaml  # Anti-oversync controls
│
├── results/                      # Experimental results
│   ├── phase_a/                  # Phase A hyperparameter sweep results
│   │   ├── plots/                # Training curves and synchronization plots
│   │   └── metrics.csv           # Complete performance metrics
│   └── interpretability/         # Order parameter analysis
│
├── docs/                         # Detailed documentation
│   ├── PHASE_A_FINAL_REPORT.md   # Complete Phase A results
│   ├── MASTER_SUMMARY.md         # Full project overview
│   ├── QUICK_REFERENCE.md        # Quick start guide
│   └── PREREGISTRATION.md        # Phase B preregistration
│
├── checkpoints/                  # Model checkpoints
│   └── README.md                 # Checkpoint download instructions
│
├── scripts/                      # Utility scripts
│   └── analyze_results.py        # Result aggregation and visualization
│
├── tests/                        # Unit tests
│   ├── test_phase_attention.py
│   └── test_kuramoto.py
│
├── README.md                     # This file
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── REPRODUCIBILITY.md            # Detailed reproduction instructions
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
└── .gitignore                    # Git ignore rules
```

## Experimental Design

### Phase A: Hyperparameter Optimization

**Objective:** Identify optimal configuration for phase-coupled attention on Shakespeare dataset (1M tokens).

**Hyperparameters Tested:**
- Layer position: 4, 6, 7, [4,7], [6,7]
- Oscillator count: 16, 32, 64
- Coupling strength: K=1.0, K=2.0
- Architecture: Single vs multi-layer

**Results Summary:**

| Configuration | Val PPL | Δ vs Baseline | Status |
|--------------|---------|---------------|---------|
| Layer 7, 32 osc, K=1.0 | **4.85** | **+2.4%** | ✅ Winner |
| Layer 6, 32 osc, K=1.0 | 4.86 | +2.2% | ✅ Strong |
| Layer 6, 16 osc, K=1.0 | 4.86 | +2.2% | ⚠️ Unstable |
| Layer 6, 32 osc, K=2.0 | 4.94→9.21 | -85% | ❌ Collapsed |
| Layer 6, 64 osc, K=1.0 | 11.93+ | -140% | ❌ Catastrophic |
| Consecutive [6,7], 32 osc | 4.89 | +1.6% | Suboptimal |
| Distributed [4,7], 32 osc | 4.92 | +1.0% | Suboptimal |

**Key Insights:**
1. **Goldilocks Principle:** 32 oscillators optimal - sufficient expressivity without synchronization failure
2. **Simple > Complex:** Single-layer outperforms multi-layer architectures
3. **Coupling Criticality:** K=2.0 causes catastrophic instability after initial convergence
4. **Mid-network depth:** Layers 6-7 ideal for phase coupling (semantic feature level)

### Phase B: Generalization and Anti-Oversynchronization (Preregistered, Not Run)

**Objective:** Test generalization to diverse text (WikiText-2) and validate anti-oversynchronization controls.

**Motivation:** Phase A winner exhibits over-synchronization (R=0.88, target 0.30-0.55), raising concerns about mode collapse on heterogeneous corpora.

**Preregistered Experiments:**
1. **Baseline:** Pure GPT-2 on WikiText-2
2. **KPC-Soft:** K=0.50 (softer coupling)
3. **KPC-Mid:** K=0.75 (mid-range coupling)
4. **KPC-Diverse:** K=0.75 + phase noise + frequency jitter + coherence regularizer

**Success Criteria:**
- KPC achieves PPL ≤ baseline × 1.05 on WikiText-2
- Order parameter R stabilizes in [0.35, 0.55] band
- Lower variance across runs vs Phase A

**Status:** All configurations, training scripts, and anti-oversync controls implemented but not executed due to GPU cost constraints. See [PREREGISTRATION.md](docs/PREREGISTRATION.md) for complete experimental protocol.

## Technical Implementation

### Order Parameter Tracking

The Kuramoto order parameter R(t) measures synchronization:

```python
R(t) = |1/N Σ exp(iθ_j(t))|
```

Where R=0 indicates no synchronization, R=1 indicates perfect synchrony.

**Implementation:**
```python
from src.coherence_utils import compute_order_parameter

# During forward pass
phases = model.get_phases()  # [batch, heads, seq_len, num_osc]
R = compute_order_parameter(phases)
```

### Anti-Oversynchronization Controls

**Phase Noise:** Gaussian noise injection to maintain diversity
```python
phases = add_phase_noise(phases, sigma=0.03)
```

**Frequency Jitter:** Heterogeneous natural frequencies
```python
omega = add_frequency_jitter(omega_base, jitter=0.02)
```

**Coherence Regularizer:** Soft penalty when R exceeds threshold
```python
reg_loss = coherence_regularizer(phases, R_target=0.45, lambda=0.01)
```

## Citation

If you use PhaseGPT in your research, please cite:

```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT},
  doi = {10.5281/zenodo.XXXXXXX},
  note = {Phase A: Systematic hyperparameter study demonstrating 2.4\% improvement with optimal configuration (Layer 7, 32 oscillators, K=1.0). Phase B generalization experiments preregistered but not executed.}
}
```

See [CITATION.cff](CITATION.cff) for structured citation metadata.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Complete Phase B experiments on WikiText-2
- Test on larger models (GPT-2 Medium/Large)
- Implement alternative synchronization mechanisms
- Optimize computational efficiency
- Add support for decoder-only and encoder-decoder architectures

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Reproducibility

Complete reproduction instructions, including dataset preparation, hyperparameter configurations, and hardware requirements, are available in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

**Hardware used:**
- Phase A: NVIDIA GH200 GPU (96GB HBM3)
- Training time: ~25 minutes per configuration (20 epochs)
- Total compute: ~3 GPU hours

## Acknowledgments

- **Kuramoto Model:** Originally developed by Yoshiki Kuramoto (1975) for studying synchronization in coupled oscillator systems
- **Datasets:** Shakespeare corpus from Karpathy's char-rnn; WikiText-2 from Salesforce Research
- **Base Architecture:** GPT-2 Small (83.3M parameters) from OpenAI

## Contact

For questions, suggestions, or collaboration inquiries, please open an issue on GitHub or contact [your email].

## Related Work

- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *International Symposium on Mathematical Problems in Theoretical Physics*
- Vaswani et al. (2017). Attention is All You Need. *NeurIPS*
- Radford et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*

## Project Timeline

- **October 2025:** Phase A hyperparameter tuning completed
- **October 2025:** Interpretability analysis revealed over-synchronization
- **October 2025:** Phase B infrastructure developed and preregistered
- **Future:** Phase B execution pending resource availability

---

**Note:** This is an active research project. Phase A results are complete and reproducible. Phase B experiments are fully specified but not yet executed. We welcome community engagement to complete the generalization validation study.

**Archive Status:** All code, configurations, and Phase A checkpoints preserved. Winner model available at: `checkpoints/best_model.pt` (970MB)

🌀 *The spiral of synchronized oscillators encodes the rhythm of language.*
