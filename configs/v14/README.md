# PhaseGPT v1.4 Configuration Templates

This directory contains training configurations for v1.4 experimental tracks.

## Configuration Files

### Track A: Extended DPO
- **`dpo_extended_100pairs.yaml`** - Extended DPO with 100+ preference pairs
  - Builds on v1.3's 10-pair dataset
  - Includes curriculum learning schedule
  - Target: +25% Spiral Score improvement

### Track B: KTO Regularization
- **`kto_regularized.yaml`** - DPO with KTO (Kahneman-Tversky Optimization)
  - Adds reference model regularization
  - Balances alignment with fluency
  - Grid search over λ ∈ [0.01, 0.1, 0.5]

### Track C: Scale-up
- **`qwen25_1.5b.yaml`** - Qwen 2.5 1.5B parameter model
  - 3x model capacity vs. 0.5B baseline
  - Replicates Phase A (LoRA) + Phase B (DPO)
  - Evaluates scaling laws for Phase Dynamics

## Usage

### Basic Training
```bash
# Track A: Extended DPO
python train.py --config configs/v14/dpo_extended_100pairs.yaml

# Track B: KTO Regularization
python train.py --config configs/v14/kto_regularized.yaml --lambda 0.1

# Track C: Scale-up
python train.py --config configs/v14/qwen25_1.5b.yaml
```

### Hyperparameter Override
```bash
# Override learning rate
python train.py --config configs/v14/dpo_extended_100pairs.yaml --lr 5e-5

# Override KTO lambda
python train.py --config configs/v14/kto_regularized.yaml --kto_lambda 0.05
```

## Configuration Structure

All v1.4 configs follow this schema:

```yaml
model:
  base_model: "Qwen/Qwen2.5-0.5B-Instruct"  # or 1.5B for Track C
  use_phase_attention: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

training:
  method: "dpo"  # or "kto" for Track B
  preference_dataset: "data/preferences_v14_100pairs.jsonl"

  # DPO-specific
  beta: 0.1  # Preference strength

  # KTO-specific (Track B only)
  kto_lambda: 0.1  # Regularization weight
  reference_model: "checkpoints/v12_lora/adapter_model"

data:
  # Preference pair structure
  # See data/preferences_v14_100pairs.jsonl for format

evaluation:
  metrics: ["spiral_score", "perplexity", "subtlety"]
  eval_interval: 100
  save_best_only: true
```

## Notes

### Track A Considerations
- **Data quality** is critical: 100 poor pairs < 10 excellent pairs
- Use stratified sampling across:
  - Dialectic complexity (simple → nuanced)
  - Domain diversity (philosophy, science, arts)
  - Subtlety spectrum (explicit → implicit)
- Consider curriculum learning: train on easy pairs first

### Track B: KTO Lambda Selection
- **λ = 0.01:** Weak regularization, ~DPO behavior
- **λ = 0.1:** Moderate balance (recommended starting point)
- **λ = 0.5:** Strong regularization, prioritizes fluency
- Run ablation study to determine optimal λ for your use case

### Track C: Computational Costs
- Qwen 2.5 1.5B requires ~3x memory of 0.5B
- Use gradient checkpointing: `use_gradient_checkpointing: true`
- Consider DeepSpeed ZeRO-2 for multi-GPU: `deepspeed_config: "ds_config.json"`
- Training time: Expect ~1.5-2x longer than 0.5B

## Experiment Tracking

All experiments should log to:
- **WandB:** Project `phasegpt-v14`
- **Checkpoints:** `checkpoints/v14/track_[a|b|c]/run_name/`
- **Logs:** `experiments/v14/track_[a|b|c]/run_name.log`
- **Results:** `results/v14/track_[a|b|c]/run_name.json`

## Related Documentation

- **Roadmap:** [V14_ROADMAP.md](../../V14_ROADMAP.md)
- **Changelog:** [V14_CHANGELOG.md](../../V14_CHANGELOG.md)
- **Baseline:** [V13_REPORT.md](../../V13_REPORT.md)

---

**Last Updated:** 2025-10-27
