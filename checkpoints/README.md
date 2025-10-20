# Model Checkpoints

This directory contains trained model checkpoints for the PhaseGPT project.

---

## Available Checkpoints

### Phase A Winner: Layer 7, 32 Oscillators, K=1.0

**Performance:**
- Validation PPL: 4.85 (2.4% improvement over baseline)
- Training time: 25 minutes (20 epochs on GH200)
- Order parameter: R = 0.8837 ± 0.0263

**Configuration:**
```yaml
model:
  type: gpt2
  n_layers: 12
  n_heads: 12
  d_model: 768
  vocab_size: 65  # char-level Shakespeare
  use_phase_attention: true
  phase_layer_idx: [7]
  num_oscillators: 32
  coupling_strength: 1.0
```

**Checkpoint Details:**
- File: `best_model.pt`
- Size: 970 MB
- Format: PyTorch state dict
- Epoch: 18 (best validation)
- Training dataset: Shakespeare (1M characters)

---

## Download Instructions

Due to GitHub file size limitations, model checkpoints are hosted externally.

### Option 1: Hugging Face Hub (Recommended)

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download checkpoint
huggingface-cli download templetwo/phasegpt-checkpoints best_model.pt \
    --local-dir checkpoints/
```

**Direct download link:**
https://huggingface.co/templetwo/phasegpt-checkpoints/resolve/main/best_model.pt

### Option 2: Zenodo

**DOI:** [10.5281/zenodo.XXXXXXX]

**Download:**
```bash
wget https://zenodo.org/record/XXXXXXX/files/best_model.pt \
     -O checkpoints/best_model.pt
```

### Option 3: OSF

**OSF Project:** https://osf.io/XXXXX/

Navigate to project → Files → Checkpoints → Download `best_model.pt`

### Option 4: Direct Link (Google Drive)

**Link:** [Google Drive Share Link]

```bash
# Using gdown
pip install gdown
gdown https://drive.google.com/uc?id=XXXXXXXXX -O checkpoints/best_model.pt
```

---

## Loading Checkpoints

### Load Full Checkpoint (with optimizer state)

```python
import torch
from src.model import GPT2Model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

# Initialize model
config = checkpoint['config']['model']
model = GPT2Model(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    n_layers=config['n_layers'],
    n_heads=config['n_heads'],
    use_phase_attention=config['use_phase_attention'],
    phase_layer_idx=config['phase_layer_idx'],
    num_oscillators=config['num_oscillators'],
    coupling_strength=config['coupling_strength'],
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
```

### Load Model Weights Only

```python
# If checkpoint is large and you only need weights
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Extract Configuration

```python
# Get configuration for reproduction
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
config = checkpoint['config']

import yaml
with open('configs/loaded_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

---

## Checkpoint Contents

**PyTorch checkpoint dictionary:**
```python
{
    'epoch': 18,                          # Training epoch
    'global_step': 3240,                  # Total training steps
    'model_state_dict': {...},            # Model weights (OrderedDict)
    'optimizer_state_dict': {...},        # AdamW optimizer state
    'scheduler_state_dict': {...},        # Learning rate scheduler
    'best_val_loss': 1.5788,              # Best validation loss (log scale)
    'best_val_ppl': 4.85,                 # Best validation perplexity
    'config': {                           # Complete configuration
        'model': {...},
        'training': {...},
        'data': {...},
    },
    'rng_state': {...},                   # Random number generator states
}
```

---

## Using Checkpoint for Inference

### Generate Text

```python
import torch
from src.model import GPT2Model
from src.data import CharDataset

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = GPT2Model(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load character mapping
dataset = CharDataset('data/shakespeare/input.txt')
char_to_idx = dataset.char_to_idx
idx_to_char = dataset.idx_to_char

# Generate text
prompt = "To be or not to be"
prompt_ids = torch.tensor([char_to_idx[c] for c in prompt]).unsqueeze(0)

with torch.no_grad():
    for _ in range(100):  # Generate 100 characters
        logits, _ = model(prompt_ids)
        next_logit = logits[0, -1, :]
        next_id = torch.multinomial(torch.softmax(next_logit, dim=-1), 1)
        prompt_ids = torch.cat([prompt_ids, next_id.unsqueeze(0)], dim=1)

generated = ''.join([idx_to_char[i.item()] for i in prompt_ids[0]])
print(generated)
```

### Analyze Synchronization

```python
# Extract order parameter from checkpoint
from PhaseB.scripts.interpret_model import analyze_checkpoint

R_stats = analyze_checkpoint(
    checkpoint_path='checkpoints/best_model.pt',
    num_tokens=512,
    device='cuda'
)

print(f"Mean R: {R_stats['R_mean']:.4f}")
print(f"R std: {R_stats['R_std']:.4f}")
```

---

## Resume Training

```python
import torch
from src.train import Trainer
from src.model import GPT2Model
from src.data import get_dataloaders

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Initialize model and optimizer
model = GPT2Model(**checkpoint['config']['model'])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Restore states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
train_loader, val_loader = get_dataloaders(...)
trainer = Trainer(...)
trainer.train(start_epoch=start_epoch, total_epochs=50)
```

---

## Checkpoint Verification

### Verify File Integrity

```bash
# Check file size
ls -lh checkpoints/best_model.pt
# Should be: ~970 MB

# Compute SHA256 hash
sha256sum checkpoints/best_model.pt
# Expected: [hash will be provided after upload]
```

### Verify Model Performance

```python
# Test on validation set
from src.evaluate import evaluate

checkpoint = torch.load('checkpoints/best_model.pt')
model = GPT2Model(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])

val_ppl = evaluate(model, val_loader, device='cuda')
print(f"Validation PPL: {val_ppl:.2f}")
# Expected: 4.85 ± 0.05
```

---

## Additional Checkpoints

### Baseline GPT-2 (No Phase Coupling)

**File:** `baseline_gpt2.pt`
**Size:** ~320 MB
**PPL:** 4.97
**Download:** [Link to be added]

### Phase A Configurations (All 7)

Available upon request. Contact maintainers if needed for reproduction studies.

---

## Creating Your Own Checkpoints

### Training from Scratch

```bash
# Train Phase A winner config
python src/train.py \
    --config configs/phase_a_winner.yaml \
    --device cuda \
    --epochs 20 \
    --save_dir checkpoints/my_run/

# Checkpoint will be saved as:
# checkpoints/my_run/best_model.pt
```

### Checkpoint Frequency

By default, checkpoints are saved:
- Every 5 epochs
- When validation improves (best model)
- At training completion (final model)

Configure in YAML:
```yaml
training:
  save_interval: 5          # Save every N epochs
  save_best: true           # Save best validation model
  save_final: true          # Save final model
```

---

## Storage Recommendations

### For Repository Maintainers

**Hugging Face Hub:**
- Pro: Integrated with `transformers` library
- Pro: Version control for models
- Pro: Free for public models
- Con: Requires account

**Zenodo:**
- Pro: DOI assignment (citable)
- Pro: Long-term preservation (CERN)
- Pro: No size limits for reasonable files
- Con: No versioning after publication

**OSF:**
- Pro: Integration with research workflow
- Pro: Component structure
- Con: 5GB file size limit per file
- Solution: Split checkpoint or compress

### For Users

**Local storage:**
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download winner model
# [Use appropriate method from above]

# Organize multiple checkpoints
checkpoints/
├── best_model.pt           # Phase A winner
├── baseline_gpt2.pt        # Baseline
└── phase_b/                # Phase B (future)
```

---

## Checkpoint Metadata

### best_model.pt Metadata

```json
{
  "model_name": "PhaseGPT Phase A Winner",
  "config": "Layer 7, 32 oscillators, K=1.0",
  "dataset": "Shakespeare (char-level)",
  "validation_ppl": 4.85,
  "order_parameter_R": 0.8837,
  "training_epochs": 18,
  "training_time_minutes": 25,
  "hardware": "NVIDIA GH200 (96GB HBM3)",
  "date_trained": "2025-10-19",
  "pytorch_version": "2.1.0",
  "cuda_version": "12.1",
  "file_size_mb": 970,
  "sha256": "[hash]",
  "license": "MIT",
  "doi": "10.5281/zenodo.XXXXXXX"
}
```

Metadata file: `checkpoints/best_model_info.json`

---

## Troubleshooting

### Issue: Out of Memory When Loading

**Solution:** Load to CPU first, then move to GPU
```python
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda')
```

### Issue: Version Mismatch Error

**Error:**
```
RuntimeError: Error(s) in loading state_dict for GPT2Model:
    Missing key(s) in state_dict: ...
```

**Solution:** Check PyTorch version
```python
import torch
print(torch.__version__)  # Should be 2.0+
```

### Issue: Slow Download

**Solutions:**
- Use `wget` with `--continue` flag for resume
- Use `aria2c` for parallel downloading
- Download from nearest mirror (if available)

---

## License

All model checkpoints are released under the MIT License, consistent with the project code.

**Citation:**
When using these checkpoints in research, please cite:

```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT},
  doi = {10.17605/OSF.IO/ZQBC4}
}
```

---

## Contact

For issues with checkpoint downloads or usage:
- GitHub Issues: https://github.com/templetwo/PhaseGPT/issues
- Email: contact@templetwo.dev

For bulk access or special requests (e.g., all Phase A checkpoints):
- Contact maintainers directly

---

**Last Updated:** 2025-10-20
**Checkpoint Version:** 1.0 (Phase A)
**Status:** Winner model available, additional checkpoints upon request
