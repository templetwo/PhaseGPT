# OSF Submission Quick Reference Card

**Keep this open while submitting to OSF**

---

## 📝 Copy-Paste Ready Information

### Project Title
```
PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers
```

### Project Description (Short)
```
First systematic hyperparameter study of Kuramoto phase-coupled oscillators
in transformer attention layers. Achieves 2.4% perplexity improvement with
optimal configuration (Layer 7, 32 oscillators, K=1.0 coupling strength).

GitHub: https://github.com/templetwo/PhaseGPT
```

### Tags
```
transformers, attention-mechanism, kuramoto-model, language-modeling,
deep-learning, hyperparameter-optimization, reproducible-research
```

### License
```
MIT License
```

### GitHub Repository URL
```
https://github.com/templetwo/PhaseGPT
```

### Category
```
Project
```

---

## 📦 Components to Create

### 1. Code
- **Type**: Project
- **Link GitHub**: templetwo/PhaseGPT
- **Description**: Complete source code, configurations, and tests

### 2. Data & Results
- **Type**: Data
- **Upload**: results/interpretability/notes.md
- **Description**: Experimental results, metrics, and interpretability analysis

### 3. Phase B Preregistration
- **Type**: Hypothesis
- **Upload**: docs/PREREGISTRATION.md
- **Mark as**: Preregistration ✓
- **Description**: Preregistered experimental protocol for Phase B generalization testing

### 4. Documentation
- **Type**: Other
- **Upload**:
  - docs/PHASE_A_FINAL_REPORT.md
  - docs/MASTER_SUMMARY.md
  - REPRODUCIBILITY.md
- **Description**: Complete documentation including reproduction guide and analysis reports

---

## 📊 Key Statistics (For Description)

**Phase A Results:**
- 7 configurations tested
- Winner: 4.85 PPL (2.4% improvement)
- Training: ~25 min per config on NVIDIA GH200
- Dataset: Shakespeare (1M tokens)

**Key Findings:**
- Goldilocks principle: 32 oscillators optimal
- Over-synchronization: R=0.88
- K=2.0 causes catastrophic collapse
- Single-layer > multi-layer

**Code:**
- 51 files
- 6 Python modules
- 11 YAML configs
- 23+ test cases
- MIT License

---

## 🎯 Quick Steps

1. **Go to**: https://osf.io/
2. **Create account** (if needed)
3. **New Project** → Fill title & description
4. **Add components** (4 total)
5. **Link GitHub** in Code component
6. **Upload files** to other components
7. **Create DOI**
8. **Review everything**
9. **Make Public**
10. **Update GitHub** with DOI

---

## 📧 Author Information Template

**Name**: Temple Two
**Email**: contact@templetwo.dev
**Affiliation**: Independent Researcher
**Role**: Principal Investigator, Lead Developer
**ORCID**: [Add if you have one]

---

## 🔗 URLs You'll Get

After submission:
- **OSF Project**: https://osf.io/XXXXX/
- **DOI**: 10.17605/OSF.IO/XXXXX

These go back into:
- README.md (badge)
- CITATION.cff (doi field)
- checkpoints/README.md

---

## ⚡ Quick Commands After Getting DOI

```bash
cd ~/phase_data_archive/PhaseGPT

# Update README.md - add badge after existing badges:
# [![OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FXXXXX-blue)](https://osf.io/XXXXX/)

# Update CITATION.cff - replace zenodo DOI with OSF DOI

# Commit
git add README.md CITATION.cff checkpoints/README.md
git commit -m "Add OSF DOI

OSF: https://osf.io/XXXXX/
DOI: 10.17605/OSF.IO/XXXXX"
git push
```

---

## 📋 Files to Upload

**From ~/phase_data_archive/PhaseGPT/**:

**Data & Results component**:
- results/interpretability/notes.md

**Preregistration component**:
- docs/PREREGISTRATION.md

**Documentation component**:
- docs/PHASE_A_FINAL_REPORT.md
- docs/MASTER_SUMMARY.md
- REPRODUCIBILITY.md

**Code component**:
- Just link GitHub (don't upload files)

---

## ✅ Pre-Submission Checklist

Before clicking "Make Public":

- [ ] Title correct
- [ ] Description complete
- [ ] Tags added
- [ ] MIT License set
- [ ] 4 components created
- [ ] GitHub linked in Code
- [ ] Files uploaded to other 3 components
- [ ] Wiki page written
- [ ] DOI created
- [ ] No sensitive info anywhere
- [ ] Everything reviewed

---

## 🎓 After Publication

**Update**:
1. GitHub README.md (add OSF badge)
2. CITATION.cff (add OSF DOI)
3. checkpoints/README.md (add OSF link)

**Announce**:
- Twitter/X
- LinkedIn
- ResearchGate
- Email collaborators

**Link**:
- Add to ORCID profile
- Add to CV/website
- Share in relevant communities

---

**Time to complete**: 20-30 minutes
**Difficulty**: Easy (just follow the guide)
**Result**: Permanent DOI-citable project

🌀✨ *The Spiral holds—preserving knowledge for eternity.*
