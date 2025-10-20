# OSF Project Metadata for PhaseGPT

This document contains structured metadata for Open Science Framework (OSF) project submission.

---

## Project Title

**PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers - A Systematic Hyperparameter Study**

---

## Alternative Titles

- Optimizing Kuramoto Phase-Coupled Attention for Language Modeling
- Synchronization Dynamics in Transformer Attention: A Hyperparameter Investigation

---

## Authors

**Primary Author:**
- Name: [Your Full Name]
- Affiliation: [Your Institution]
- ORCID: [Your ORCID ID]
- Email: [Your Email]
- Role: Principal Investigator, Lead Developer
- Contribution: Conceptualization, Implementation, Experimentation, Analysis, Writing

**Additional Contributors** (if applicable):
- Name: [Advisor/Collaborator Name]
- Affiliation: [Institution]
- ORCID: [ORCID if available]
- Role: Advisor/Collaborator
- Contribution: Supervision, Methodology Guidance

---

## Project Abstract

This project presents the first systematic hyperparameter study of Kuramoto phase-coupled oscillator mechanisms integrated into transformer attention layers for language modeling. Phase A experiments systematically evaluated 7 configurations across 4 hyperparameter dimensions (layer position, oscillator count, coupling strength, architecture type) on the Shakespeare dataset. The optimal configuration (single layer at position 7, 32 oscillators, coupling strength K=1.0) achieved 4.85 perplexity, representing a 2.4% improvement over baseline GPT-2.

**Key Contributions:**

1. **Goldilocks Principle Discovery:** 32 oscillators optimal—16 oscillators show instability, 64 oscillators cause catastrophic synchronization failure
2. **Coupling Criticality:** K=2.0 causes catastrophic training collapse (PPL degrades from 4.94 to 9.21 after epoch 20)
3. **Over-Synchronization Paradox:** Winner configuration exhibits high order parameter (R=0.88, target 0.30-0.55), raising generalization concerns
4. **Architecture Simplicity:** Single-layer phase attention outperforms multi-layer designs

Phase B generalization experiments on WikiText-2 with anti-oversynchronization controls are fully preregistered and implemented but not executed due to computational resource constraints. The complete codebase, configurations, trained checkpoints, and experimental protocols are publicly archived to enable reproduction and community continuation.

---

## Keywords

- Transformer architectures
- Attention mechanisms
- Kuramoto model
- Phase-coupled oscillators
- Synchronization dynamics
- Language modeling
- Neural networks
- Hyperparameter optimization
- Interpretability
- Order parameter
- Coherence analysis
- Deep learning
- Natural language processing
- Computational neuroscience
- Dynamical systems

---

## Research Area / Discipline

**Primary:** Computer Science - Machine Learning

**Secondary:**
- Artificial Intelligence
- Natural Language Processing
- Computational Neuroscience
- Nonlinear Dynamics
- Complex Systems

**ACM Classification:**
- Computing methodologies → Neural networks
- Computing methodologies → Natural language processing
- Theory of computation → Dynamic systems

---

## Project Description (Detailed)

### Background

Transformer architectures have revolutionized natural language processing through self-attention mechanisms. Recent work has explored integrating dynamical systems principles into neural networks, with limited investigation of synchronization phenomena. The Kuramoto model, developed by Yoshiki Kuramoto in 1975, describes spontaneous synchronization in coupled oscillator systems and has explained diverse phenomena from firefly flashing to neuronal activity.

This project investigates whether Kuramoto phase-coupled dynamics can provide inductive biases beneficial for language modeling by enabling attention heads to coordinate through phase synchronization rather than purely learned weights.

### Methodology

**Phase A: Hyperparameter Optimization**
- Model: GPT-2 Small (83.3M parameters)
- Dataset: Shakespeare corpus (1M tokens, character-level)
- Hardware: NVIDIA GH200 GPU (96GB HBM3)
- Training: 20 epochs per configuration, batch size 32 (effective)
- Baseline: Standard GPT-2 achieving 4.97 perplexity

**Hyperparameters Explored:**
1. Layer position: 4, 6, 7, [4,7] consecutive, [6,7] distributed
2. Oscillator count: 16, 32, 64
3. Coupling strength: K=1.0, K=2.0
4. Architecture: Single-layer vs multi-layer phase attention

**Phase B: Generalization Testing (Preregistered, Not Executed)**
- Dataset: WikiText-2 (word-level, 2M tokens)
- Configurations: Baseline, KPC-soft (K=0.50), KPC-mid (K=0.75), KPC-diverse (K=0.75 + anti-oversync controls)
- Anti-oversync mechanisms: Phase noise (σ=0.03), frequency jitter (2%), coherence regularization

### Results Summary

**Optimal Configuration:** Layer 7, 32 oscillators, K=1.0
- Validation PPL: 4.85 (2.4% improvement)
- Training time: 25 minutes (20 epochs)
- Order parameter: R=0.8837 (over-synchronized)

**Critical Findings:**
- Catastrophic failure at 64 oscillators (PPL > 11.93)
- Training collapse at K=2.0 coupling
- Multi-layer architectures underperform single-layer
- Over-synchronization despite strong narrow-corpus performance

### Reproducibility

All experiments are fully reproducible:
- Complete source code (MIT licensed)
- All configuration files
- Trained model checkpoints (970MB winner model)
- Training logs and metrics
- Detailed documentation

### Open Questions

Phase B experiments would address:
1. Does over-synchronization harm generalization to diverse text?
2. Can anti-oversync controls achieve target R range (0.35-0.55)?
3. What is the optimal coupling strength for general language modeling?

---

## Data Availability Statement

### Datasets Used

**Shakespeare Corpus:**
- Source: Karpathy's char-rnn repository
- Size: ~1MB, 1M characters
- License: Public domain
- Availability: Included in repository under `data/shakespeare/`

**WikiText-2 (Phase B, not executed):**
- Source: Salesforce Research
- Size: ~12MB, 2M tokens
- License: Creative Commons Attribution-ShareAlike
- Availability: Downloadable via Hugging Face `datasets` library

### Code and Model Availability

**Code:**
- Repository: GitHub (https://github.com/yourusername/PhaseGPT)
- License: MIT
- Complete implementation including all experiments

**Model Checkpoints:**
- Winner model: 970MB checkpoint
- Storage: [Hugging Face Hub / Zenodo / Google Drive]
- DOI: [To be assigned upon OSF publication]

**Experimental Artifacts:**
- Training logs (all configurations)
- Validation metrics (CSV format)
- Order parameter trajectories
- Configuration files (YAML)

All data, code, and models are permanently archived and publicly accessible.

---

## Funding Information

**Funding Status:** [Select one]
- [ ] Funded by [Grant Agency, Grant Number]
- [x] Unfunded independent research
- [ ] Institutional support from [Institution]

**Computational Resources:**
- Cloud GPU rental: Lambda Labs NVIDIA GH200 instance
- Total cost: ~$30 USD (3 GPU hours at $10/hour)
- Self-funded

---

## Preregistration Status

**Phase A:** Not preregistered (exploratory hyperparameter search)

**Phase B:** Preregistered on 2025-10-20
- Preregistration document: `docs/PREREGISTRATION.md`
- Hypotheses, methods, and success criteria specified before execution
- Status: Not executed (infrastructure complete, experiments not run)

---

## Ethical Considerations

**Data Ethics:**
- Shakespeare corpus: Public domain, no privacy concerns
- WikiText-2: Curated dataset, no identifiable information

**Research Ethics:**
- Open science principles followed
- Negative results (catastrophic failures) reported
- Limitations clearly documented
- Computational costs disclosed

**Environmental Impact:**
- Total compute: ~3 GPU hours (Phase A)
- Estimated CO2: ~1.5 kg (based on Lambda Labs energy mix)
- Efficient training: 25 minutes per configuration

---

## Related Publications

**Prior Work:**
- None (original research)

**Planned Publications:**
1. Conference paper: "Optimizing Kuramoto Phase-Coupled Attention: A Hyperparameter Study"
   - Target venue: ICLR 2026 or NeurIPS 2026 Workshop
   - Status: In preparation (Phase A complete)

2. Extended journal article: "Synchronization Dynamics in Transformer Attention" (pending Phase B completion)

---

## License Information

**Code License:** MIT License
- Permissive open-source license
- Commercial use allowed
- Attribution required

**Documentation License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Data License:**
- Shakespeare: Public domain
- WikiText-2: CC BY-SA (as per original)

---

## Project Category

- [x] Data
- [x] Code/Software
- [x] Materials/Protocols
- [ ] Analysis Scripts
- [x] Preregistration
- [x] Supplementary Materials

---

## Project Tags

```
#transformers #attention #Kuramoto #oscillators #synchronization
#language-modeling #deep-learning #NLP #hyperparameter-tuning
#interpretability #open-science #reproducibility #preregistration
```

---

## DOI and Persistent Identifiers

**OSF Project DOI:** [Will be assigned upon OSF publication]
- Format: 10.17605/OSF.IO/XXXXX

**Zenodo Archive DOI:** [Optional, for snapshot]
- Format: 10.5281/zenodo.XXXXXXX

**GitHub Repository:** https://github.com/yourusername/PhaseGPT
- Permanent archive via Software Heritage

---

## Version Information

**Current Version:** 1.0.0
**Release Date:** 2025-10-20

**Version History:**
- v1.0.0 (2025-10-20): Initial release with Phase A complete, Phase B preregistered

---

## Contact Information

**Primary Contact:**
- Name: [Your Name]
- Email: [Your Email]
- Institution: [Your Institution]
- Website: [Your Website/Lab Page]

**Project Homepage:** https://github.com/yourusername/PhaseGPT

**Issues/Questions:** Submit via GitHub Issues

---

## Acknowledgments

**Theoretical Foundations:**
- Yoshiki Kuramoto: Original Kuramoto model (1975)

**Datasets:**
- Andrej Karpathy: Shakespeare corpus preparation
- Salesforce Research: WikiText-2 dataset

**Infrastructure:**
- Lambda Labs: GPU cloud computing platform
- OpenAI: GPT-2 base architecture

**Community:**
- Open-source contributors (see CONTRIBUTING.md)

---

## Supplementary Materials

The following materials are included in the OSF project:

1. **Complete Source Code** (`src/`)
2. **Experiment Configurations** (`configs/`)
3. **Phase A Results** (`results/phase_a/`)
4. **Detailed Reports** (`docs/`)
5. **Training Logs** (archived separately due to size)
6. **Model Checkpoints** (link to external storage)
7. **Preregistration Document** (`docs/PREREGISTRATION.md`)
8. **Reproducibility Instructions** (`REPRODUCIBILITY.md`)

---

## Project Timeline

- **2025-10-15:** Project initialization
- **2025-10-16 - 2025-10-19:** Phase A hyperparameter experiments
- **2025-10-19:** Phase A completion, optimal configuration identified
- **2025-10-20:** Interpretability analysis (over-synchronization discovered)
- **2025-10-20:** Phase B preregistration
- **2025-10-20:** Public archive and OSF submission
- **Future:** Phase B execution (pending resources)

---

## Usage Instructions for OSF Submission

1. **Create OSF Project:** Visit https://osf.io/ and create new project
2. **Copy Information:** Use fields above to populate OSF metadata forms
3. **Upload Files:**
   - Code repository (clone from GitHub)
   - Documentation files
   - Configuration files
   - Link to checkpoint storage
4. **Set Visibility:** Make public upon Phase A publication
5. **Obtain DOI:** Generate DOI for citation
6. **Update Repository:** Add OSF DOI to README.md and CITATION.cff

---

**Document Version:** 1.0
**Last Updated:** 2025-10-20
**Purpose:** OSF project metadata for PhaseGPT publication
