# 🌀 PhaseGPT - START HERE

**Welcome to PhaseGPT!** This document is your entry point to the complete PhaseGPT project.

**Last Updated**: 2025-10-20
**Status**: ✅ **PUBLICATION READY**
**Repository Size**: 436KB (51 files, 17 markdown docs)
**GitHub**: https://github.com/templetwo/PhaseGPT

---

## 🎯 Quick Navigation

**Choose your path**:

| Goal | Document | Time |
|------|----------|------|
| **Get started fast** | [QUICKSTART.md](QUICKSTART.md) | 5 min |
| **Publish to GitHub** | [GETTING_STARTED.md](GETTING_STARTED.md) | 1-2 hours |
| **Reproduce experiments** | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | 3-4 hours |
| **Understand research** | [docs/PHASE_A_FINAL_REPORT.md](docs/PHASE_A_FINAL_REPORT.md) | 30 min |
| **Publication checklist** | [REPOSITORY_READY.md](REPOSITORY_READY.md) | Reference |
| **Complete overview** | [docs/MASTER_SUMMARY.md](docs/MASTER_SUMMARY.md) | 20 min |

---

## 📚 Documentation Map

### For Users & Experimenters
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to training
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - Complete reproduction guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute

### For Publishers
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - GitHub publication guide
- **[REPOSITORY_READY.md](REPOSITORY_READY.md)** - Publication checklist
- **[docs/OSF_METADATA.md](docs/OSF_METADATA.md)** - OSF submission info

### For Researchers
- **[docs/PHASE_A_FINAL_REPORT.md](docs/PHASE_A_FINAL_REPORT.md)** - Complete Phase A results
- **[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)** - Phase B protocol
- **[docs/MASTER_SUMMARY.md](docs/MASTER_SUMMARY.md)** - Project archive

### Reference
- **[README.md](README.md)** - Main project README
- **[CITATION.cff](CITATION.cff)** - Citation metadata
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick reference card

---

## 🚀 Quick Commands

### Setup
```bash
# Clone repository
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT

# Install dependencies
pip install -r requirements.txt

# Download data
bash data/shakespeare/download.sh
```

### Train Winner Configuration
```bash
python src/train.py --config configs/phase_a_winner.yaml
```

### Run Tests
```bash
pytest tests/ -v
```

### Publish to GitHub
```bash
./setup_git.sh  # Follow prompts
```

---

## 🎓 Research Summary

**PhaseGPT** presents the first systematic hyperparameter study of Kuramoto phase-coupled oscillators in transformer attention layers.

### Key Results

**Phase A (Complete)**:
- ✅ 7 configurations systematically tested
- ✅ **Winner**: Layer 7, 32 oscillators, K=1.0 → **4.85 PPL** (2.4% improvement)
- ✅ **Goldilocks principle** discovered: 32 oscillators optimal
- ✅ **Over-synchronization paradox**: R=0.88 achieved strong performance
- ✅ **Coupling instability**: K=2.0 causes catastrophic collapse

**Phase B (Preregistered, Not Run)**:
- 🔄 Infrastructure complete
- 🔄 4 WikiText-2 configurations ready
- 🔄 Anti-oversynchronization controls implemented
- ⏸️ Experiments paused due to GPU resource constraints

### Scientific Contributions

1. **First systematic study** of Kuramoto oscillators in transformers
2. **Goldilocks principle** for oscillator count (32 optimal)
3. **Over-synchronization paradox** identified and documented
4. **Catastrophic collapse** at K=2.0 coupling strength

---

## 📊 Repository Statistics

**Files**:
- 51 total files
- 17 markdown documentation files
- 6 Python source modules
- 11 YAML configuration files
- 3 test modules

**Size**:
- Repository: 436KB
- Winner checkpoint: 970MB (hosted externally)
- Complete archive: 876MB

**Tests**:
- 3 test modules
- 23+ test cases
- Full pytest compatibility

---

## ✅ What's Complete

### Implementation
✅ Complete phase-coupled attention mechanism
✅ Order parameter (R) tracking integrated
✅ Anti-oversynchronization controls (noise, jitter, regularization)
✅ Full return_info infrastructure for interpretability
✅ All 7 Phase A configurations documented
✅ All 4 Phase B configurations ready

### Documentation
✅ Professional README with badges
✅ Complete Phase A results report
✅ Phase B preregistration protocol
✅ Reproducibility guide
✅ Contributing guidelines
✅ OSF metadata prepared
✅ Publication checklist

### Infrastructure
✅ Python package structure (`src/__init__.py`)
✅ Comprehensive test suite (pytest-ready)
✅ Data download scripts (Shakespeare, WikiText-2)
✅ Git initialization script
✅ GitHub configuration templates

### Metadata
✅ MIT License
✅ Structured CITATION.cff
✅ requirements.txt
✅ environment.yml
✅ Proper .gitignore

---

## 🎯 Next Actions

### Immediate (Required for Publication)

1. **Create GitHub repository**
   - Run: `./setup_git.sh`
   - Follow prompts

2. **Upload checkpoint**
   ```bash
   huggingface-cli upload templetwo/phasegpt-checkpoints \
       ~/phase_data_archive/phase_a_implementation/runs/.../best_model.pt \
       best_model.pt
   ```

3. **Create release v1.0.0**
   - See [GETTING_STARTED.md](GETTING_STARTED.md#creating-a-release-v100)

### Optional (Enhance Visibility)

4. **Submit to OSF**
   - Create project at osf.io
   - Link GitHub repository
   - Request DOI

5. **Announce on social media**
   - Template in [GETTING_STARTED.md](GETTING_STARTED.md#announce-your-work)

6. **Submit to Papers with Code**
   - Add implementation link

---

## 💡 Common Questions

### Q: Where do I start?
**A**: See [QUICKSTART.md](QUICKSTART.md) for fastest path to running experiments.

### Q: How do I publish to GitHub?
**A**: Run `./setup_git.sh` and follow prompts. Complete guide in [GETTING_STARTED.md](GETTING_STARTED.md).

### Q: Can I reproduce the results?
**A**: Yes! Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Expected: 4.85 PPL ± 0.05.

### Q: Where are the checkpoints?
**A**: Winner checkpoint (970MB) needs to be uploaded to Hugging Face. Instructions in [checkpoints/README.md](checkpoints/README.md).

### Q: What about Phase B?
**A**: Infrastructure complete but experiments not run. See [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) for protocol.

### Q: How do I contribute?
**A**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🌟 Highlights

**What makes PhaseGPT special?**

1. **First systematic study** of Kuramoto oscillators in transformers
2. **Complete reproducibility** - every experiment documented
3. **Novel findings** - Goldilocks principle, over-synchronization paradox
4. **Publication-ready** - all materials prepared for GitHub and OSF
5. **Preregistered** - Phase B experiments fully specified before execution
6. **Open science** - MIT license, FAIR principles, community engagement

---

## 📞 Support

**Documentation**: See [docs/](docs/) folder
**Issues**: https://github.com/templetwo/PhaseGPT/issues
**Email**: contact@templetwo.dev
**Citation**: See [CITATION.cff](CITATION.cff)

---

## 🏁 Ready to Publish?

**3-Step Quick Publish**:

1. **Setup Git**: `./setup_git.sh`
2. **Upload checkpoint**: Hugging Face CLI
3. **Create release**: GitHub v1.0.0

**Time**: ~1-2 hours total

See [GETTING_STARTED.md](GETTING_STARTED.md) for complete walkthrough.

---

**🌀✨ The Spiral holds the pattern. All knowledge preserved and ready to share.**

---

**Project Status**: ✅ COMPLETE & PUBLICATION-READY
**Date**: 2025-10-20
**Version**: 1.0.0
**GitHub**: https://github.com/templetwo/PhaseGPT
