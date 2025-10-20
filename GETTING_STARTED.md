# Getting Started with PhaseGPT

**Complete guide from repository setup to publication** 📚

---

## 🎯 Where to Start?

Choose your path based on your goals:

### Path A: Quick Experimentation (30 minutes)
→ See **[QUICKSTART.md](QUICKSTART.md)**

### Path B: Full Reproduction (3-4 hours)
→ See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**

### Path C: Publishing to GitHub (1-2 hours)
→ **You're in the right place!** Continue below.

### Path D: Understanding the Research (1 hour)
→ See **[docs/PHASE_A_FINAL_REPORT.md](docs/PHASE_A_FINAL_REPORT.md)**

---

## 📦 Publishing PhaseGPT to GitHub

This guide walks you through publishing the repository to GitHub.

### Prerequisites

✅ You have the PhaseGPT repository at: `~/phase_data_archive/PhaseGPT/`
✅ You have a GitHub account (username: `templetwo`)
✅ Git is installed on your system
✅ You're ready to make the repository public

### Step 1: Navigate to Repository

```bash
cd ~/phase_data_archive/PhaseGPT
```

### Step 2: Run Automated Setup Script

We've created a script that handles everything:

```bash
./setup_git.sh
```

This script will:
1. Initialize Git repository
2. Add all files
3. Create initial commit with detailed message
4. Set up GitHub remote
5. Push to GitHub

**Follow the prompts** - the script will guide you through each step.

### Step 3: Create GitHub Repository (If Not Done)

If you haven't created the repository yet:

1. Go to https://github.com/new
2. **Repository name**: `PhaseGPT`
3. **Description**: `Kuramoto Phase-Coupled Oscillator Attention in Transformers`
4. **Visibility**: Public
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **"Create repository"**

Then return to Step 2 and run `./setup_git.sh` again.

### Step 4: Configure GitHub Repository Settings

#### Add Topics/Tags

Go to: https://github.com/templetwo/PhaseGPT

Click the ⚙️ icon next to "About" and add topics:
- `transformers`
- `attention-mechanism`
- `kuramoto-model`
- `language-modeling`
- `deep-learning`
- `research`
- `reproducible-research`

#### Update Description

Set description to:
```
Systematic investigation of Kuramoto phase-coupled oscillator mechanisms in transformer attention. Achieves 2.4% perplexity improvement with optimal config (Layer 7, 32 osc, K=1.0).
```

#### Enable GitHub Pages (Optional)

Settings → Pages → Source: Deploy from main branch, /docs folder

This will host your documentation at: `https://templetwo.github.io/PhaseGPT/`

---

## 🏷️ Creating a Release (v1.0.0)

After pushing to GitHub, create an official release:

### Tag the Release

```bash
cd ~/phase_data_archive/PhaseGPT

git tag -a v1.0.0 -m "Phase A complete: Optimal configuration identified

Key findings:
- Layer 7, 32 oscillators, K=1.0 → 4.85 PPL (2.4% improvement)
- Goldilocks principle: 32 oscillators optimal
- Over-synchronization discovered: R=0.88
- K=2.0 coupling causes catastrophic collapse

Phase B infrastructure ready but not executed."

git push origin v1.0.0
```

### Create Release on GitHub

1. Go to: https://github.com/templetwo/PhaseGPT/releases
2. Click **"Draft a new release"**
3. Choose tag: `v1.0.0`
4. Release title: `PhaseGPT v1.0.0 - Phase A Complete`
5. Description:

```markdown
## PhaseGPT v1.0.0 - Phase A Complete

First systematic hyperparameter study of Kuramoto phase-coupled oscillators in transformers.

### Key Findings

**Performance:**
- **2.4% improvement** in perplexity (4.85 vs 4.97 baseline)
- Optimal: Layer 7, 32 oscillators, K=1.0 coupling

**Novel Discoveries:**
- **Goldilocks principle**: 32 oscillators optimal (16 unstable, 64 catastrophic)
- **Over-synchronization paradox**: R=0.88 achieved strong performance on narrow corpus
- **Coupling instability**: K=2.0 causes catastrophic collapse (9.21 PPL)

### What's Included

✅ Complete source code with phase attention mechanism
✅ 11 configuration files (7 Phase A + 4 Phase B)
✅ Comprehensive test suite (23+ test cases)
✅ Full documentation and reproduction guide
✅ Preregistered Phase B experiments (not run)

### Installation

```bash
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT
pip install -r requirements.txt
bash data/shakespeare/download.sh
```

### Quick Start

```bash
# Train Phase A winner
python src/train.py --config configs/phase_a_winner.yaml
```

See [QUICKSTART.md](QUICKSTART.md) for complete guide.

### Citation

```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT},
  version = {1.0.0}
}
```

### Documentation

- [Complete results](docs/PHASE_A_FINAL_REPORT.md)
- [Reproduction guide](REPRODUCIBILITY.md)
- [Project overview](docs/MASTER_SUMMARY.md)

### Future Work

Phase B generalization experiments are preregistered but not executed due to resource constraints. See [PREREGISTRATION.md](docs/PREREGISTRATION.md) for complete experimental protocol.

🌀 *The spiral of synchronized oscillators encodes the rhythm of language.*
```

6. Click **"Publish release"**

---

## 📤 Upload Checkpoint to Hugging Face

The winner checkpoint (~970MB) is too large for GitHub.

### Create Hugging Face Repository

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (you'll need a Hugging Face account)
huggingface-cli login

# Create model repository
huggingface-cli repo create phasegpt-checkpoints --type model
```

### Upload Checkpoint

```bash
# Upload winner checkpoint
huggingface-cli upload templetwo/phasegpt-checkpoints \
    ~/phase_data_archive/phase_a_implementation/runs/gpt2-small_20251019_211620/checkpoints/best_model.pt \
    best_model.pt

# Verify upload
echo "Checkpoint uploaded to: https://huggingface.co/templetwo/phasegpt-checkpoints"
```

### Create Model Card

On Hugging Face, create a README.md for the model:

```markdown
# PhaseGPT Checkpoints

Model checkpoints for PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers.

## Phase A Winner: Layer 7, 32 Oscillators, K=1.0

**Performance:**
- Validation PPL: 4.85 (2.4% improvement over baseline)
- Order parameter: R = 0.8837 ± 0.0263

**Configuration:**
- Model: GPT-2 Small (83.3M parameters)
- Phase attention: Layer 7 only
- Oscillators: 32
- Coupling strength: K=1.0
- Dataset: Shakespeare (char-level)

**Usage:**
```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="templetwo/phasegpt-checkpoints",
    filename="best_model.pt"
)
```

**Citation:**
```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT}
}
```

**License:** MIT

**Repository:** https://github.com/templetwo/PhaseGPT
```

---

## 🔬 Submit to Open Science Framework (OSF)

### Create OSF Project

1. Go to https://osf.io/ and sign in
2. Click **"Create new project"**
3. **Title**: `PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers`
4. **Category**: Project
5. Click **"Create"**

### Add Components

Add the following components to organize materials:

#### 1. Code Component
- **Type**: Component
- **Title**: "Code"
- **Link**: Connect to GitHub repository
- Go to: Settings → Add-ons → GitHub → Link Repository → `templetwo/PhaseGPT`

#### 2. Data Component
- **Title**: "Experimental Results"
- Upload: `results/phase_a/` directory
- Upload: `results/interpretability/notes.md`

#### 3. Materials Component
- **Title**: "Configurations"
- Upload all files from: `configs/`

#### 4. Preregistration Component
- **Title**: "Phase B Preregistration"
- Upload: `docs/PREREGISTRATION.md`
- **Important**: Mark this as a preregistration (before Phase B execution)

### Fill Project Metadata

Use information from `docs/OSF_METADATA.md`:

**Description:**
```
This project presents the first systematic hyperparameter study of Kuramoto
phase-coupled oscillators in transformer attention layers for language modeling.

Key Findings:
- 2.4% perplexity improvement with optimal configuration
- Goldilocks principle for oscillator count (32 optimal)
- Over-synchronization paradox discovered
- Catastrophic collapse documented for K=2.0 coupling

Phase A (Complete): 7 configurations tested on Shakespeare dataset
Phase B (Preregistered): WikiText-2 generalization experiments
```

**Tags:**
- transformers
- attention mechanism
- Kuramoto model
- language modeling
- hyperparameter optimization
- reproducible research

### Request DOI

1. In your OSF project, click **"Create DOI"**
2. Once assigned, update:
   - `README.md` (add DOI badge)
   - `CITATION.cff` (add DOI)
   - `checkpoints/README.md` (add DOI)

### Make Public

1. Review all components
2. Click **"Make Public"** button
3. Confirm

---

## 📣 Announce Your Work

### Social Media Template

```
🌀 Excited to share PhaseGPT: First systematic study of Kuramoto
phase-coupled oscillators in transformer attention!

Key findings:
✓ 2.4% perplexity improvement
✓ "Goldilocks principle" for oscillator count
✓ Over-synchronization paradox discovered

Code, checkpoints, and complete reproduction guide:
https://github.com/templetwo/PhaseGPT

#MachineLearning #Transformers #Research #OpenScience
```

### Academic Mailing Lists

Consider posting to:
- NeurIPS mailing list
- ICLR discussion forums
- r/MachineLearning on Reddit
- Papers with Code

---

## ✅ Publication Checklist

Use this checklist to track your progress:

- [ ] GitHub repository created and public
- [ ] All code pushed to main branch
- [ ] Release v1.0.0 created
- [ ] Topics/tags added
- [ ] Repository description updated
- [ ] Checkpoint uploaded to Hugging Face
- [ ] Model card created
- [ ] OSF project created and public
- [ ] DOI assigned from OSF
- [ ] README.md updated with DOI badge
- [ ] CITATION.cff updated with DOI
- [ ] Announcement posted on social media
- [ ] GitHub repository URL shared

---

## 🎓 Next Steps

### If Continuing Research

1. **Run Phase B experiments** (if GPU available)
   - Complete WikiText-2 sweep
   - Test anti-oversynchronization controls
   - Update results

2. **Write paper**
   - Use Phase A report as foundation
   - Include or note Phase B status
   - Target: NeurIPS workshop or ICLR

3. **Expand project**
   - Test on GPT-2 Medium/Large
   - Try other datasets
   - Implement optimizations

### If Publishing Only

1. **Monitor repository**
   - Respond to issues within 48 hours
   - Review pull requests
   - Engage with community

2. **Track citations**
   - Set up Google Scholar alert
   - Update README with papers citing

3. **Maintain documentation**
   - Fix typos/errors as reported
   - Add FAQ section if needed
   - Update based on feedback

---

## 📞 Getting Help

**Questions about publication process?**
- Check: [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)
- Check: [REPOSITORY_READY.md](REPOSITORY_READY.md)

**Questions about the research?**
- Read: [docs/PHASE_A_FINAL_REPORT.md](docs/PHASE_A_FINAL_REPORT.md)
- Read: [docs/MASTER_SUMMARY.md](docs/MASTER_SUMMARY.md)

**Technical issues?**
- Check: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Open issue: https://github.com/templetwo/PhaseGPT/issues

---

**Congratulations on preparing PhaseGPT for publication!** 🎉

Your research is now ready to be shared with the world. The complete infrastructure for reproducible, open science is in place.

🌀✨ *The Spiral holds the pattern. All knowledge shared.*
