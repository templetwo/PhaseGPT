# 🔬 PhaseGPT OSF Submission Guide

**Step-by-step guide to submit PhaseGPT to the Open Science Framework**

**Estimated Time**: 20-30 minutes
**Result**: DOI-citable research project

---

## 📋 Pre-Submission Checklist

Before starting, ensure you have:

- ✅ OSF account (create at https://osf.io/register if needed)
- ✅ GitHub repository published (https://github.com/templetwo/PhaseGPT)
- ✅ ORCID ID (optional but recommended - get at https://orcid.org/)
- ✅ This guide open for reference

---

## Step 1: Create OSF Account (If Needed)

**Go to**: https://osf.io/register

**Fill in**:
- Full name
- Email address
- Create password

**Verify email** and you're ready!

---

## Step 2: Create New OSF Project

### 2.1 Start Project Creation

**Go to**: https://osf.io/
**Click**: "Create new project" (green button)

### 2.2 Fill Project Information

**Title**:
```
PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers
```

**Storage Location**:
- United States (or your region)

**Category**:
- Project

**Description** (copy this):
```
First systematic hyperparameter study of Kuramoto phase-coupled oscillators
in transformer attention layers. Achieves 2.4% perplexity improvement with
optimal configuration (Layer 7, 32 oscillators, K=1.0 coupling strength).

Key Findings:
• Goldilocks principle: 32 oscillators optimal (16 unstable, 64 catastrophic)
• Over-synchronization paradox identified (R=0.88)
• Coupling instability at K=2.0 causes catastrophic collapse
• Single-layer design outperforms multi-layer architectures

Complete code, configurations, and reproduction guide at:
https://github.com/templetwo/PhaseGPT

Phase A: Complete (7 configurations tested)
Phase B: Preregistered but not executed (resource constraints)
```

**Tags**:
- transformers
- attention-mechanism
- kuramoto-model
- language-modeling
- deep-learning
- hyperparameter-optimization
- reproducible-research

**Click**: "Create"

---

## Step 3: Configure Project Settings

### 3.1 Add License

**Click**: "Settings" → "License"
**Select**: MIT License
**Save**

### 3.2 Add Contributors (If Applicable)

**Click**: "Contributors" → "Add"
**Enter**: Email addresses of collaborators
**Set permissions**: Read, Write, or Admin

---

## Step 4: Create Project Components

OSF organizes materials into components. Let's create 4:

### Component 1: Code

**Click**: "Add Component"
- **Title**: Code
- **Category**: Project
- **Description**: Complete source code, configurations, and tests

**After creating, add GitHub link**:
1. Click into the "Code" component
2. Click "Settings" → "Add-ons"
3. Select "GitHub"
4. Connect account
5. Select repository: templetwo/PhaseGPT
6. Save

### Component 2: Data & Results

**Click**: "Add Component"
- **Title**: Data & Results
- **Category**: Data
- **Description**: Experimental results, metrics, and interpretability analysis

**Upload files** (in Files tab):
1. Click "Upload"
2. Select from `~/phase_data_archive/PhaseGPT/results/interpretability/notes.md`
3. Can add more later

### Component 3: Preregistration

**Click**: "Add Component"
- **Title**: Phase B Preregistration
- **Category**: Hypothesis
- **Description**: Preregistered experimental protocol for Phase B generalization testing

**Upload**:
- `~/phase_data_archive/PhaseGPT/docs/PREREGISTRATION.md`

**Important**: Check "This is a preregistration" option

### Component 4: Documentation

**Click**: "Add Component"
- **Title**: Documentation
- **Category**: Other
- **Description**: Complete documentation including reproduction guide and analysis reports

**Upload**:
- docs/PHASE_A_FINAL_REPORT.md
- docs/MASTER_SUMMARY.md
- REPRODUCIBILITY.md

---

## Step 5: Add Detailed Metadata

### 5.1 Project Wiki (Overview Page)

**Click**: "Wiki" → "Edit"

**Content** (copy this):

```markdown
# PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers

## Overview

First systematic hyperparameter study of Kuramoto phase-coupled oscillators integrated
into transformer attention layers for language modeling.

## Key Results

**Phase A Complete:**
- 7 configurations systematically tested on Shakespeare dataset (1M tokens)
- Winner: Layer 7, 32 oscillators, K=1.0 → **4.85 PPL** (2.4% improvement)
- Training time: ~25 minutes per config on NVIDIA GH200

**Novel Discoveries:**
1. **Goldilocks Principle**: 32 oscillators optimal (16 unstable, 64 catastrophic)
2. **Over-Synchronization Paradox**: R=0.88 achieved strong performance but raises
   generalization concerns
3. **Coupling Instability**: K=2.0 causes catastrophic collapse (PPL 4.94 → 9.21)
4. **Architecture Simplicity**: Single-layer outperforms multi-layer designs

**Phase B Status:**
- Infrastructure complete with anti-oversynchronization controls
- 4 WikiText-2 configurations ready
- Preregistered but not executed (GPU resource constraints)

## Repository

**GitHub**: https://github.com/templetwo/PhaseGPT
**License**: MIT
**Version**: 1.0.0

## Components

- **Code**: Complete implementation with GitHub integration
- **Data & Results**: Experimental metrics and interpretability analysis
- **Preregistration**: Phase B experimental protocol
- **Documentation**: Reproduction guide and detailed reports

## Citation

```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT},
  doi = {[Will be added after OSF DOI assignment]}
}
```

## Reproducibility

Complete reproduction guide available in repository. Expected results:
- Validation PPL: 4.85 ± 0.05
- Training time: ~25 minutes on GH200 GPU
- All 7 Phase A configurations reproducible

## Contact

- GitHub Issues: https://github.com/templetwo/PhaseGPT/issues
- Email: contact@templetwo.dev
```

**Save**

### 5.2 Update Project Description

**Click**: Main project → "Edit" (pencil icon near title)

Add full abstract from OSF_METADATA.md (already done in step 2.2)

---

## Step 6: Request DOI

### 6.1 Create DOI

**Click**: "Settings" (in left sidebar of main project)
**Scroll to**: "Create DOI"
**Click**: "Create DOI"

**Confirmation**: Read and confirm
- You understand DOI is permanent
- Project metadata is correct
- You're ready to make public

**Click**: "Create"

### 6.2 Copy DOI

**Your DOI will look like**: `10.17605/OSF.IO/XXXXX`

**Save this DOI** - you'll need to update files!

---

## Step 7: Make Project Public

⚠️ **Important**: Review everything before this step!

### 7.1 Final Review Checklist

- [ ] Project title correct
- [ ] Description complete
- [ ] All components created
- [ ] GitHub linked in Code component
- [ ] Files uploaded to appropriate components
- [ ] License set to MIT
- [ ] Wiki page complete
- [ ] DOI created
- [ ] No sensitive information anywhere

### 7.2 Make Public

**Click**: "Make Public" button (top right)

**Confirmation dialog**:
- [ ] I understand this project will be publicly visible
- [ ] I confirm all information is accurate
- [ ] I have reviewed all components

**Click**: "Confirm"

🎉 **Your project is now public!**

---

## Step 8: Update Repository with OSF DOI

Now that you have the DOI, update your GitHub repository:

### 8.1 Update README.md

Add DOI badge at the top:

```markdown
[![OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FXXXXX-blue)](https://osf.io/XXXXX/)
```

### 8.2 Update CITATION.cff

```bash
cd ~/phase_data_archive/PhaseGPT
```

Edit CITATION.cff, replace:
```yaml
doi: "10.5281/zenodo.XXXXXXX"
```

With:
```yaml
doi: "10.17605/OSF.IO/XXXXX"  # Your actual DOI
```

And add:
```yaml
  - type: doi
    value: "10.17605/OSF.IO/XXXXX"
    description: "OSF project DOI"
```

### 8.3 Update Other Files

Files that need the DOI:
- checkpoints/README.md
- docs/OSF_METADATA.md

### 8.4 Commit and Push

```bash
cd ~/phase_data_archive/PhaseGPT

git add README.md CITATION.cff checkpoints/README.md docs/OSF_METADATA.md
git commit -m "Add OSF DOI to documentation

OSF project: https://osf.io/XXXXX/
DOI: 10.17605/OSF.IO/XXXXX"
git push
```

---

## Step 9: Cross-Link GitHub and OSF

### On GitHub

Add OSF link to repository description:
```
OSF Project: https://osf.io/XXXXX/
```

### On OSF

Already done via GitHub integration in Code component!

---

## Step 10: Announce and Share

### Social Media Post Template

```
🎉 PhaseGPT is now published on OSF with a permanent DOI!

📊 First systematic study of Kuramoto oscillators in transformers
✅ Complete code, data, and reproduction guide
🔬 Preregistered Phase B experiments
📖 Full documentation and analysis

OSF: https://osf.io/XXXXX/
GitHub: https://github.com/templetwo/PhaseGPT
DOI: 10.17605/OSF.IO/XXXXX

#OpenScience #MachineLearning #Reproducibility
```

### Where to Share

- Twitter/X
- LinkedIn
- ResearchGate
- Your institutional page
- Relevant mailing lists (NeurIPS, ICLR, etc.)

---

## 📊 Your Complete URLs

After completion, you'll have:

**Primary Resources**:
- **GitHub**: https://github.com/templetwo/PhaseGPT
- **OSF Project**: https://osf.io/XXXXX/ (your actual ID)
- **DOI**: 10.17605/OSF.IO/XXXXX (permanent citation)

**Components**:
- **Code**: https://osf.io/XXXXX/wiki/Code/
- **Data**: https://osf.io/XXXXX/wiki/Data%20&%20Results/
- **Preregistration**: https://osf.io/XXXXX/wiki/Phase%20B%20Preregistration/
- **Documentation**: https://osf.io/XXXXX/wiki/Documentation/

---

## 🎓 Benefits of OSF Publication

**Citability**:
- ✅ Permanent DOI (never changes)
- ✅ Structured citation metadata
- ✅ Discoverable via DOI search engines

**Preservation**:
- ✅ Long-term archival (OSF committed to preservation)
- ✅ Version control for project updates
- ✅ Backup of all materials

**Discoverability**:
- ✅ Indexed by Google Scholar
- ✅ Searchable on OSF platform
- ✅ Connected to ORCID profile (if linked)

**Credibility**:
- ✅ Timestamps all materials (proves priority)
- ✅ Preregistration support (demonstrates rigor)
- ✅ Open science standard compliance

---

## 🐛 Troubleshooting

### Issue: Can't create DOI

**Solution**: Ensure project is complete with title, description, and at least one component with content.

### Issue: GitHub integration not working

**Solution**:
1. Disconnect and reconnect GitHub account
2. Ensure repository is public
3. Try refreshing the connection

### Issue: Files won't upload

**Solution**:
1. Check file size (OSF has 5GB limit per file)
2. Try uploading smaller files first
3. Use component structure to organize

### Issue: DOI not showing

**Solution**: Wait 5-10 minutes after creation. If still not visible, contact OSF support.

---

## 📞 Getting Help

**OSF Support**: support@osf.io
**Documentation**: https://help.osf.io/
**Status Page**: https://status.cos.io/

---

## ✅ Final Checklist

After completing all steps, you should have:

- [ ] OSF project created and public
- [ ] DOI assigned (10.17605/OSF.IO/XXXXX)
- [ ] GitHub repository linked in Code component
- [ ] All 4 components created with appropriate files
- [ ] Project wiki complete
- [ ] MIT License applied
- [ ] DOI added to GitHub repository files
- [ ] Cross-links between GitHub and OSF
- [ ] Announcement prepared
- [ ] ORCID profile updated (if applicable)

---

**🌀✨ Congratulations! PhaseGPT is now a permanently archived, DOI-citable research project!**

Your work is preserved for posterity and discoverable by researchers worldwide.

---

**Created**: 2025-10-20
**For**: PhaseGPT v1.0.0
**Repository**: https://github.com/templetwo/PhaseGPT
