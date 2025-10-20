# Shakespeare Dataset

Character-level dataset from Shakespeare's works, commonly used for language modeling benchmarks.

## Download

```bash
bash download.sh
```

This will download `input.txt` (~1MB, ~1M characters) from Karpathy's char-rnn repository.

## Dataset Details

- **Source**: Combined works of William Shakespeare
- **Size**: ~1M characters
- **Vocabulary**: 65 unique characters (letters, punctuation, spaces)
- **Format**: Plain text, character-level
- **Splits**: 90% train, 10% validation (configured in YAML)

## Usage

The dataset is automatically loaded by the training script when using Shakespeare configs:

```bash
python src/train.py --config configs/phase_a_winner.yaml
```

## Citation

If using this dataset, cite the original source:

```
Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
```

## Expected Results

With the Phase A winner configuration (Layer 7, 32 osc, K=1.0):
- **Validation PPL**: ~4.85
- **Training time**: ~25 minutes on GH200 GPU (20 epochs)
- **Baseline PPL**: 4.97 (no phase coupling)
