# PhaseGPT Quick Start Guide

**Get up and running in 5 minutes** ⚡

---

## 🚀 Fastest Path to Results

### 1. Clone Repository

```bash
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT
```

### 2. Install Dependencies

```bash
# Option A: pip (fast)
pip install -r requirements.txt

# Option B: conda (isolated)
conda env create -f environment.yml
conda activate phasegpt
```

### 3. Download Data

```bash
bash data/shakespeare/download.sh
```

### 4. Train Phase A Winner

```bash
python src/train.py --config configs/phase_a_winner.yaml --device cuda --epochs 20
```

**Expected Results**:
- Validation PPL: ~4.85 (2.4% improvement over baseline)
- Training time: ~25 minutes on GH200 GPU
- Order parameter R: ~0.88

---

## 📊 Reproduce All Phase A Experiments

```bash
# Run all 7 configurations
for config in configs/phase_a/*.yaml; do
    echo "Training $(basename $config)..."
    python src/train.py --config $config --device cuda --epochs 20
done
```

**Time**: ~3 hours total (7 configs × 25 min)

---

## 🔬 Analyze Synchronization

```bash
# Download pre-trained checkpoint (if available)
huggingface-cli download templetwo/phasegpt-checkpoints best_model.pt \
    --local-dir checkpoints/

# Run interpretability analysis
python scripts/interpret_model.py \
    --checkpoint checkpoints/best_model.pt \
    --num_tokens 512 \
    --output_dir results/interpretability/
```

**Output**: Order parameter R statistics and visualization

---

## 🧪 Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

## 📈 Compare Configurations

```bash
# Train baseline (no phase coupling)
python src/train.py --config configs/baseline.yaml --device cuda --epochs 164

# Train Phase A winner (with phase coupling)
python src/train.py --config configs/phase_a_winner.yaml --device cuda --epochs 20

# Compare results
python scripts/analyze_results.py --baseline baseline --phase_a phase_a_winner
```

---

## 🎯 Key Results Summary

| Configuration | Val PPL | Status |
|--------------|---------|--------|
| **Layer 7, 32 osc, K=1.0** | **4.85** | ✅ **WINNER** |
| Layer 6, 32 osc, K=1.0 | 4.86 | ✅ Strong |
| Baseline (no KPC) | 4.97 | Baseline |
| Layer 6, 32 osc, K=2.0 | 9.21 | ❌ Collapsed |
| Layer 6, 64 osc, K=1.0 | 11.93+ | ❌ Catastrophic |

---

## 📖 What's Next?

### Option 1: Dive Deeper into Documentation
- **Complete results**: `docs/PHASE_A_FINAL_REPORT.md`
- **Reproduction guide**: `REPRODUCIBILITY.md`
- **Project overview**: `docs/MASTER_SUMMARY.md`

### Option 2: Run Phase B Experiments (If GPU Available)
```bash
# Test generalization on WikiText-2
python scripts/train_generalize.py --config configs/phase_b/wt2_baseline.yaml
python scripts/train_generalize.py --config configs/phase_b/wt2_kpc_soft.yaml
```

### Option 3: Contribute
- See `CONTRIBUTING.md` for guidelines
- Areas for contribution:
  - Complete Phase B experiments
  - Test on larger models (GPT-2 Medium/Large)
  - Optimize computational efficiency
  - Add visualization tools

---

## ⚙️ Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB
- Storage: 5GB

**Recommended** (for full reproduction):
- GPU: NVIDIA A100/H100 or GH200
- RAM: 32GB+
- Storage: 20GB

**CPU-only** (slow, for testing):
```bash
python src/train.py --config configs/phase_a_winner.yaml --device cpu --epochs 2
```

---

## 🐛 Common Issues

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size
```yaml
# Edit config YAML
training:
  batch_size: 8  # Reduce from 32
```

### Issue 2: Package Not Found

**Solution**: Install from requirements
```bash
pip install -r requirements.txt --upgrade
```

### Issue 3: Dataset Not Found

**Solution**: Download Shakespeare data
```bash
bash data/shakespeare/download.sh
```

### Issue 4: Slow Training

**Solution**: Check GPU utilization
```bash
nvidia-smi  # Should show GPU in use
# If CPU-bound, check data loading:
# - Increase num_workers in data loader
# - Use pin_memory=True
```

---

## 📞 Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: https://github.com/templetwo/PhaseGPT/issues
- **Discussions**: GitHub Discussions tab
- **Email**: contact@templetwo.dev

---

## 🎓 Citation

If you use PhaseGPT in your research:

```bibtex
@software{phasegpt2025,
  title = {PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers},
  author = {Temple Two},
  year = {2025},
  url = {https://github.com/templetwo/PhaseGPT},
  note = {Phase A: 2.4\% improvement with Layer 7, 32 osc, K=1.0}
}
```

---

## ✨ Quick Command Reference

```bash
# Setup
git clone https://github.com/templetwo/PhaseGPT.git
pip install -r requirements.txt
bash data/shakespeare/download.sh

# Train winner
python src/train.py --config configs/phase_a_winner.yaml

# Run tests
pytest tests/ -v

# Analyze synchronization
python scripts/interpret_model.py --checkpoint checkpoints/best_model.pt

# Generate text
python src/evaluate.py --checkpoint checkpoints/best_model.pt --generate

# Resume from checkpoint
python src/train.py --config configs/phase_a_winner.yaml --resume checkpoints/best_model.pt
```

---

**Happy experimenting!** 🌀✨

For complete documentation, see **README.md** and **docs/**.
