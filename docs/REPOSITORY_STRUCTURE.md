# PhaseGPT Repository Structure Proposal

This document outlines the recommended directory organization for GitHub publication and OSF submission.

---

## Proposed Directory Structure

```
PhaseGPT/
│
├── README.md                         # Main project overview (comprehensive)
├── LICENSE                           # MIT License
├── CITATION.cff                      # Citation metadata (GitHub integration)
├── CONTRIBUTING.md                   # Contribution guidelines
├── REPRODUCIBILITY.md                # Detailed reproduction instructions
├── requirements.txt                  # Python dependencies (pip)
├── environment.yml                   # Conda environment specification
├── .gitignore                        # Git ignore rules
├── .gitattributes                    # Git LFS configuration (for checkpoints)
├── setup.py                          # Package installation script
│
├── src/                              # Core implementation
│   ├── __init__.py
│   ├── model.py                      # GPT2Model with phase attention support
│   ├── phase_attention.py            # PhaseAttention mechanism (Kuramoto)
│   ├── coherence_utils.py            # Order parameter, regularization, tracking
│   ├── train.py                      # Training loop with R(t) logging
│   ├── evaluate.py                   # Evaluation and perplexity calculation
│   └── data.py                       # Dataset loading (Shakespeare, WikiText-2)
│
├── configs/                          # Experiment configurations (YAML)
│   ├── README.md                     # Config documentation
│   ├── baseline.yaml                 # Pure GPT-2 baseline
│   ├── phase_a_winner.yaml           # Optimal config (Layer 7, 32 osc, K=1.0)
│   ├── phase_a/                      # All Phase A configs
│   │   ├── layer6_32osc_k1.0.yaml
│   │   ├── layer7_32osc_k1.0.yaml
│   │   ├── layer6_16osc_k1.0.yaml
│   │   ├── layer6_64osc_k1.0.yaml
│   │   ├── layer6_32osc_k2.0.yaml
│   │   ├── consecutive_6_7_32osc.yaml
│   │   └── distributed_4_7_32osc.yaml
│   └── phase_b/                      # Phase B configs (preregistered)
│       ├── wt2_baseline.yaml
│       ├── wt2_kpc_soft.yaml
│       ├── wt2_kpc_mid.yaml
│       └── wt2_kpc_diverse.yaml
│
├── scripts/                          # Utility and analysis scripts
│   ├── README.md
│   ├── train_generalize.py           # Phase B training script (WikiText-2)
│   ├── interpret_model.py            # Order parameter analysis
│   ├── visualize_attention.py        # Attention pattern visualization
│   ├── analyze_results.py            # Result aggregation and plotting
│   └── reproduce_phase_a.sh          # Batch reproduction script
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_phase_attention.py       # PhaseAttention mechanism tests
│   ├── test_kuramoto.py              # Kuramoto dynamics tests
│   ├── test_coherence_utils.py       # Order parameter tests
│   ├── test_model.py                 # GPT2Model tests
│   └── test_data.py                  # Dataset loading tests
│
├── data/                             # Datasets (not in Git, download scripts)
│   ├── README.md                     # Data download instructions
│   ├── shakespeare/
│   │   ├── download.sh               # Auto-download script
│   │   └── .gitignore                # Ignore data files
│   └── wikitext2/
│       ├── download.sh
│       └── .gitignore
│
├── checkpoints/                      # Model checkpoints (Git LFS or external)
│   ├── README.md                     # Download instructions for trained models
│   ├── best_model_info.json          # Metadata (size, hash, download URL)
│   └── .gitignore                    # Ignore .pt files (too large for Git)
│
├── results/                          # Experimental results
│   ├── README.md
│   ├── phase_a/
│   │   ├── metrics.csv               # Complete Phase A results table
│   │   ├── expected_metrics.csv      # For reproduction verification
│   │   ├── plots/
│   │   │   ├── ppl_curves.png        # Validation PPL trajectories
│   │   │   ├── comparison_table.png
│   │   │   └── training_dynamics.png
│   │   └── logs/                     # Archived training logs
│   │       └── .gitignore            # Ignore large log files
│   ├── interpretability/
│   │   ├── R_statistics.json         # Order parameter statistics
│   │   ├── R_trajectory.png          # R(t) plot
│   │   └── phase_heatmap.png         # Oscillator phase visualization
│   └── phase_b/                      # Placeholder for Phase B results
│       └── .gitkeep
│
├── docs/                             # Detailed documentation
│   ├── README.md                     # Documentation index
│   ├── MASTER_SUMMARY.md             # Complete project overview (archive)
│   ├── QUICK_REFERENCE.md            # Quick start guide
│   ├── PHASE_A_FINAL_REPORT.md       # Detailed Phase A results
│   ├── PREREGISTRATION.md            # Phase B preregistration
│   ├── OSF_METADATA.md               # OSF submission metadata
│   ├── figures/                      # Documentation figures
│   │   ├── architecture_diagram.png
│   │   ├── kuramoto_schematic.png
│   │   └── results_summary.png
│   └── api/                          # API documentation (Sphinx)
│       ├── conf.py
│       ├── index.rst
│       └── modules.rst
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_explore_shakespeare.ipynb
│   ├── 02_analyze_phase_a_results.ipynb
│   ├── 03_visualize_synchronization.ipynb
│   └── 04_attention_patterns.ipynb
│
├── experiments/                      # Experiment tracking (optional)
│   ├── phase_a/
│   │   └── experiment_log.md         # Dated log of experiments
│   └── phase_b/
│       └── .gitkeep
│
└── .github/                          # GitHub-specific files
    ├── workflows/
    │   ├── tests.yml                 # CI: Run pytest on push
    │   ├── linting.yml               # CI: Black, flake8, mypy
    │   └── docs.yml                  # CI: Build Sphinx docs
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── reproduction_issue.md
    └── PULL_REQUEST_TEMPLATE.md
```

---

## File Organization Principles

### 1. Separation of Concerns

**Code** (`src/`): Pure implementation, no experiment-specific logic
**Configs** (`configs/`): All hyperparameters externalized
**Scripts** (`scripts/`): Experiment orchestration and analysis
**Results** (`results/`): Output artifacts, version-controlled metadata only
**Docs** (`docs/`): Human-readable documentation

### 2. Reproducibility

**Tests** (`tests/`): Verify correctness
**Configs** (`configs/`): Exact experiment specifications
**Data scripts** (`data/`): Automated dataset preparation
**Environment** (`environment.yml`): Fixed dependencies

### 3. FAIR Principles

**Findable:**
- Clear README.md
- Descriptive file names
- GitHub topics/tags
- OSF project linking

**Accessible:**
- Public repository (GitHub)
- Open license (MIT)
- Checkpoint download links
- No authentication barriers

**Interoperable:**
- Standard formats (YAML, JSON, CSV, PNG)
- PyTorch checkpoints (widely compatible)
- Python 3.8+ (broad compatibility)

**Reusable:**
- Complete documentation
- Modular code design
- Unit tests
- Example configs

---

## File Mapping from Current Archive

### Migration Plan

**From current archive to proposed structure:**

```bash
# Source code (already organized)
cp -r phase_a_implementation/src/* src/

# Phase A configs (create from report)
# Generate configs based on PHASE_A_FINAL_REPORT.md

# Phase B configs (already ready)
cp PhaseB/configs/*.yaml configs/phase_b/

# Scripts
cp PhaseB/scripts/train_generalize.py scripts/
cp PhaseB/scripts/interpret_model.py scripts/

# Documentation
cp MASTER_SUMMARY.md docs/
cp QUICK_REFERENCE.md docs/
cp phase_a_implementation/PHASE_A_FINAL_REPORT.md docs/

# Results (create summaries)
# Extract metrics from logs into results/phase_a/metrics.csv

# Checkpoints (link to external storage)
# Upload best_model.pt to Hugging Face or Zenodo
# Create checkpoints/README.md with download link
```

---

## Large File Handling

### Git LFS (Recommended for Checkpoints)

**Setup:**
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"
git add .gitattributes
```

**Upload checkpoints:**
```bash
git add checkpoints/best_model.pt
git commit -m "Add Phase A winner checkpoint"
git push
```

**Note:** GitHub LFS has 1GB free storage, 1GB free bandwidth/month. Winner model is 970MB (fits).

### Alternative: External Storage

**Option 1: Hugging Face Hub**
```bash
huggingface-cli upload yourusername/phasegpt-checkpoints ./checkpoints/best_model.pt
```

**Option 2: Zenodo**
- Upload via web interface
- Get DOI
- Link in checkpoints/README.md

**Option 3: Google Drive / Dropbox**
- Public share link
- Include in checkpoints/README.md

---

## Data Organization

### Dataset Download Scripts

**data/shakespeare/download.sh:**
```bash
#!/bin/bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -O data/shakespeare/input.txt
echo "Shakespeare dataset downloaded successfully"
```

**data/wikitext2/download.py:**
```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("data/wikitext2/")
print("WikiText-2 dataset downloaded successfully")
```

### .gitignore in data/

```
# data/.gitignore
*.txt
*.csv
*.json
wikitext2/
shakespeare/input.txt
```

---

## Result Artifacts

### What to Version Control

**Include in Git:**
- Metadata files (metrics.csv, statistics.json)
- Small plots (<1MB, PNG format)
- Configuration files used for each experiment
- Aggregated results and summaries

**Exclude from Git:**
- Individual training logs (>10MB each)
- Raw TensorBoard events
- Large checkpoint files
- Intermediate outputs

### results/phase_a/metrics.csv Format

```csv
config,layer,num_osc,coupling_k,best_val_ppl,best_epoch,train_time_min,R_mean,R_std
phase_a_winner,7,32,1.0,4.85,18,25,0.8837,0.0263
layer6_32osc_k1.0,6,32,1.0,4.86,18,24,0.8821,0.0271
layer6_16osc_k1.0,6,16,1.0,4.86,20,23,0.7245,0.0512
layer6_64osc_k1.0,6,64,1.0,11.93,10,28,0.2134,0.1245
layer6_32osc_k2.0,6,32,2.0,9.21,36,45,0.9512,0.0087
consecutive_6_7,6-7,32,1.0,4.89,19,26,0.8934,0.0198
distributed_4_7,4-7,32,1.0,4.92,20,27,0.8123,0.0324
```

---

## Documentation Structure

### docs/ Organization

**High-level:**
- README.md → Documentation index
- MASTER_SUMMARY.md → Complete project archive
- QUICK_REFERENCE.md → Quick start guide

**Experiment-specific:**
- PHASE_A_FINAL_REPORT.md → Detailed Phase A results
- PREREGISTRATION.md → Phase B protocol

**Submission:**
- OSF_METADATA.md → OSF project metadata
- REPRODUCIBILITY.md → Reproduction guide

**API (future):**
- api/ → Sphinx-generated API docs

---

## Testing Structure

### tests/ Organization

**Unit tests:**
- test_phase_attention.py → PhaseAttention correctness
- test_kuramoto.py → Kuramoto dynamics
- test_coherence_utils.py → Order parameter, regularization

**Integration tests:**
- test_model.py → Full GPT2Model with phase attention
- test_training.py → Training loop (mini run)

**Data tests:**
- test_data.py → Dataset loading and preprocessing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_phase_attention.py::test_order_parameter -v
```

---

## GitHub-Specific Configuration

### .github/workflows/tests.yml (CI)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - run: pip install -r requirements.txt
    - run: pytest tests/ -v
```

### Issue Templates

Provide templates for:
- Bug reports (reproduction steps, environment details)
- Feature requests (use case, proposed implementation)
- Reproduction issues (hardware, error logs)

---

## Setup.py for Package Installation

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="phasegpt",
    version="1.0.0",
    description="Kuramoto Phase-Coupled Oscillator Attention in Transformers",
    author="[Your Name]",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        # ... see requirements.txt
    ],
    python_requires=">=3.8",
)
```

**Installation:**
```bash
pip install -e .  # Editable install for development
```

---

## Migration Checklist

### Pre-Publication

- [ ] Organize files per proposed structure
- [ ] Create all config files (based on Phase A experiments)
- [ ] Generate results/phase_a/metrics.csv
- [ ] Upload checkpoints to external storage
- [ ] Create checkpoint download README
- [ ] Write data download scripts
- [ ] Generate plots for docs/figures/
- [ ] Create .gitignore and .gitattributes
- [ ] Setup GitHub repository
- [ ] Initialize Git LFS (if using)

### During Migration

- [ ] Copy src/ files
- [ ] Copy config files
- [ ] Copy documentation to docs/
- [ ] Create scripts/ directory
- [ ] Add tests/
- [ ] Generate requirements.txt and environment.yml
- [ ] Create README.md, CONTRIBUTING.md, etc.
- [ ] Add LICENSE file

### Post-Migration

- [ ] Verify Git structure: `tree -L 2`
- [ ] Test installation: `pip install -e .`
- [ ] Run tests: `pytest tests/ -v`
- [ ] Build docs: `sphinx-build docs/api docs/_build`
- [ ] Test reproduction: Follow REPRODUCIBILITY.md
- [ ] Push to GitHub
- [ ] Create release v1.0.0
- [ ] Submit to OSF
- [ ] Generate DOI

---

## OSF Organization

**OSF Project Structure:**

```
PhaseGPT OSF Project
├── Code (link to GitHub)
├── Data
│   ├── Phase A Results (metrics.csv, plots)
│   └── Checkpoints (link to Hugging Face/Zenodo)
├── Materials
│   ├── Configurations (all YAML files)
│   └── Documentation (PDFs of markdown docs)
├── Preregistration
│   └── Phase B Preregistration (PREREGISTRATION.md)
└── Supplemental Materials
    ├── Training Logs (archived)
    └── Additional Figures
```

---

## Version Control Strategy

### Git Branching

- `main`: Stable, publication-ready code
- `develop`: Active development
- `feature/xyz`: New features
- `experiment/phase-b`: Phase B experiments (when started)

### Releases

- `v1.0.0`: Initial publication (Phase A complete)
- `v1.1.0`: Phase B results (when available)
- `v2.0.0`: Major architectural changes (future)

### Commit Message Convention

```
type(scope): Brief description

Detailed explanation

- Bullet points for changes
```

**Types:** feat, fix, docs, refactor, test, perf

---

## Summary

This structure optimizes for:
- **Reproducibility:** Clear separation, complete configs
- **Discoverability:** Intuitive organization, good README
- **Extensibility:** Modular design, easy to add experiments
- **Collaboration:** Tests, docs, contribution guidelines
- **Publication:** FAIR principles, OSF compatibility

**Next steps:** Execute migration plan and test full reproduction workflow.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-20
**Purpose:** Guide repository organization for GitHub and OSF publication
