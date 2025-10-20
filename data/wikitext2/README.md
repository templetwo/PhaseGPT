# WikiText-2 Dataset

Word-level language modeling dataset extracted from Wikipedia articles.

## Download

```bash
bash download.sh
```

Or use the Hugging Face datasets library directly:

```python
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
```

## Dataset Details

- **Source**: Wikipedia articles (curated by Salesforce Research)
- **Size**: ~2M tokens
- **Vocabulary**: ~33K unique words
- **Format**: Raw text (preserves original capitalization, punctuation)
- **Splits**: Train, validation, test (pre-defined)

## Usage

The dataset is automatically loaded by Phase B generalization scripts:

```bash
python src/train.py --config configs/phase_b/wt2_baseline.yaml
```

## Phase B Experiments (Preregistered, Not Run)

Four configurations designed to test generalization and anti-oversynchronization:

1. **Baseline** (`wt2_baseline.yaml`): Pure GPT-2, no phase coupling
2. **KPC-Soft** (`wt2_kpc_soft.yaml`): K=0.50, softer coupling
3. **KPC-Mid** (`wt2_kpc_mid.yaml`): K=0.75, mid-range coupling
4. **KPC-Diverse** (`wt2_kpc_diverse.yaml`): K=0.75 + noise + jitter + regularization

### Success Criteria

- KPC achieves Val PPL ≤ baseline × 1.05
- Order parameter R stabilizes in [0.35, 0.55] band
- Lower variance across runs vs Phase A

### Status

Phase B infrastructure is complete but experiments have not been run due to GPU resource constraints. See `docs/PREREGISTRATION.md` for complete experimental protocol.

## Citation

```bibtex
@article{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv preprint arXiv:1609.07843},
  year={2016}
}
```

## Expected Results (Phase B - Not Tested)

Hypothesis: Phase A winner's over-synchronization (R=0.88) may hurt performance on diverse text.

Anti-oversync controls should maintain R in [0.35, 0.55] band while achieving competitive PPL on WikiText-2.
