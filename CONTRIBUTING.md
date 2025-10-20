# Contributing to PhaseGPT

Thank you for your interest in contributing to PhaseGPT! This project explores Kuramoto phase-coupled oscillator attention mechanisms in transformers, and we welcome contributions from the research community.

## Ways to Contribute

### High-Priority Contributions

1. **Phase B Execution**: Run the preregistered WikiText-2 generalization experiments
   - All configurations ready in `configs/phase_b/`
   - Training script available at `PhaseB/scripts/train_generalize.py`
   - Estimated compute: 8-12 GPU hours
   - See [PREREGISTRATION.md](docs/PREREGISTRATION.md) for protocol

2. **Computational Efficiency**: Optimize phase computation for larger models
   - Current bottleneck: `phase_diff` tensor scales as O(seq_len²)
   - Target: Reduce memory footprint for GPT-2 Medium/Large

3. **Alternative Synchronization Mechanisms**:
   - Sakaguchi-Kuramoto model (phase-amplitude coupling)
   - Adaptive coupling strength (learnable K)
   - Hierarchical oscillator structures

### General Contributions

- Bug fixes and code improvements
- Documentation enhancements
- Unit test coverage expansion
- Visualization tools for phase dynamics
- Support for additional datasets
- Performance benchmarking on diverse architectures

## Getting Started

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/PhaseGPT.git
cd PhaseGPT

# Create conda environment
conda env create -f environment.yml
conda activate phasegpt

# Install in editable mode
pip install -e .

# Run tests
pytest tests/
```

### Project Structure

```
PhaseGPT/
├── src/                    # Core implementation
│   ├── model.py            # GPT-2 base model
│   ├── phase_attention.py  # Phase-coupled attention
│   └── coherence_utils.py  # Synchronization utilities
├── tests/                  # Unit tests (expand these!)
├── configs/                # Experiment configurations
└── docs/                   # Documentation
```

## Contribution Guidelines

### Code Style

We follow standard Python conventions:

- **PEP 8** style guide
- **Type hints** for all function signatures
- **Docstrings** in NumPy format
- **Maximum line length**: 100 characters

Example:
```python
def compute_order_parameter(
    phases: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute Kuramoto order parameter R(t).

    Parameters
    ----------
    phases : torch.Tensor
        Phase values, shape [..., num_oscillators]
    dim : int, default=-1
        Dimension along which to compute order parameter

    Returns
    -------
    R : torch.Tensor
        Order parameter values in [0, 1], shape [...]

    Notes
    -----
    R = |⟨exp(iθ)⟩| where ⟨·⟩ denotes average over oscillators.
    R=0: no synchronization, R=1: perfect synchrony.
    """
    complex_phases = torch.exp(1j * phases)
    mean_phase = torch.mean(complex_phases, dim=dim)
    R = torch.abs(mean_phase)
    return R
```

### Testing

All new features require corresponding tests:

```python
# tests/test_phase_attention.py
def test_order_parameter_bounds():
    """Order parameter should be in [0, 1]."""
    phases = torch.randn(8, 12, 32)  # [batch, heads, oscillators]
    R = compute_order_parameter(phases, dim=-1)

    assert torch.all(R >= 0.0)
    assert torch.all(R <= 1.0)

def test_synchronized_phases():
    """Identical phases should give R=1."""
    phases = torch.ones(8, 12, 32) * np.pi / 4
    R = compute_order_parameter(phases, dim=-1)

    assert torch.allclose(R, torch.ones_like(R), atol=1e-6)
```

Run tests before submitting:
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

### Commit Messages

Use clear, descriptive commit messages following conventional commits:

```
type(scope): Brief description

Detailed explanation of changes.

- List specific changes
- Reference issues: Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring without functional changes
- `test`: Adding or updating tests
- `perf`: Performance improvements

**Example:**
```
feat(phase_attention): Add adaptive coupling strength

Implement learnable K parameter that adjusts during training
to maintain target synchronization level R_target=0.45.

- Add AdaptiveCoupling module
- Update PhaseAttention to support adaptive mode
- Add unit tests for coupling adaptation
- Update config schema

Addresses #42
```

### Pull Request Process

1. **Create a branch**: `git checkout -b feature/your-feature-name`

2. **Make changes**:
   - Write clear, documented code
   - Add tests for new functionality
   - Update relevant documentation

3. **Test locally**:
   ```bash
   pytest tests/ -v
   python src/train.py --config configs/test_config.yaml --epochs 2
   ```

4. **Submit PR**:
   - Provide clear description of changes
   - Link related issues
   - Include test results
   - Add screenshots/plots if relevant

5. **Code review**: Address reviewer feedback promptly

6. **Merge**: Maintainers will merge once approved

## Specific Contribution Opportunities

### 1. Complete Phase B Experiments

**Goal**: Execute preregistered WikiText-2 experiments to test generalization hypothesis.

**Requirements**:
- GPU with 40GB+ VRAM (or run sequentially with batch_size=8)
- 8-12 hours compute time
- Follow exact protocol in [PREREGISTRATION.md](docs/PREREGISTRATION.md)

**Deliverables**:
- Training logs for all 4 configurations
- Validation PPL results
- R(t) trajectories
- Comparison plots vs baseline

**Impact**: Completes the research narrative and validates/refutes over-synchronization hypothesis.

### 2. Optimize Phase Computation

**Current bottleneck**:
```python
# phase_attention.py line ~85
phase_diff = synced_phases.unsqueeze(-2) - synced_phases.unsqueeze(-3)
# Shape: [batch, heads, seq_len, seq_len, num_osc] ≈ 12GB for batch=32
```

**Optimization approaches**:
- Chunked computation
- Flash attention integration
- Sparse phase coupling (only nearby tokens)
- Gradient checkpointing

**Success criteria**: Run GPT-2 Large with batch_size=32 on 40GB GPU.

### 3. Visualization Tools

**Needed**:
- Interactive phase trajectory plots (plotly/bokeh)
- Real-time R(t) monitoring dashboard
- Attention pattern comparisons (baseline vs phase-coupled)
- Phase space projections

**Example**:
```python
# scripts/visualize_phases.py
def plot_phase_trajectory(
    checkpoint_path: str,
    text_sample: str,
    output_path: str
):
    """Generate animated phase evolution during text processing."""
    # Load model, tokenize text, extract phases at each step
    # Create animation showing oscillator phases evolving
    pass
```

### 4. Scaling Studies

**Experiments to run**:
- GPT-2 Medium (345M params) with optimal Phase A config
- GPT-2 Large (774M params) with optimized phase computation
- Test on larger datasets (BookCorpus, C4)

**Hypothesis**: Phase coupling benefits may increase with model scale.

### 5. Architecture Variants

**Implementations to explore**:
- Encoder-decoder with phase coupling
- Cross-attention phase synchronization
- Layer-wise coupling (information flow between layers)
- Multi-head phase diversity (different K per head)

## Research Ethics

This project follows open science principles:

- **Preregistration**: Phase B experiments preregistered before execution
- **Reproducibility**: All code, configs, and data publicly available
- **Transparency**: Report negative results and limitations
- **Collaboration**: Share compute resources and insights

## Questions?

- **General questions**: Open a GitHub issue
- **Research collaboration**: Contact maintainers directly
- **Bug reports**: Use issue template
- **Feature requests**: Open discussion in GitHub Discussions

## Recognition

Contributors will be acknowledged in:
- Project README
- Academic publications (for substantial contributions)
- Release notes

Thank you for helping advance research in phase-coupled attention mechanisms!

---

**Note**: For Phase B experiments, coordinate with maintainers first to ensure protocol adherence and avoid duplicate effort.
