import sys
sys.path.insert(0, '/home/ubuntu/phase_a_implementation/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from dataclasses import fields

from model import GPT2Model, GPT2Config

def compute_order_parameter(phases):
    '''
    Compute Kuramoto order parameter R(t).
    R = |mean(exp(i*theta))|
    '''
    if isinstance(phases, np.ndarray):
        phases = torch.tensor(phases)
    complex_phases = torch.exp(1j * phases)
    R = torch.abs(complex_phases.mean(dim=-1))
    return R.cpu().numpy() if torch.is_tensor(R) else R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--sample_tokens', type=int, default=512)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint from {args.ckpt}...')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    
    # Extract model config and filter to GPT2Config fields
    config_dict_raw = ckpt['config']['model']
    valid_fields = {f.name for f in fields(GPT2Config)}
    config_dict = {k: v for k, v in config_dict_raw.items() if k in valid_fields}
    
    # Infer vocab_size from embedding if not in config
    if config_dict.get('vocab_size') is None:
        vocab_size = ckpt['model_state_dict']['token_embedding.weight'].shape[0]
        config_dict['vocab_size'] = vocab_size
        print(f'Inferred vocab_size from embedding: {vocab_size}')
    
    # Extract phase attention settings
    use_phase_attention = config_dict_raw.get('use_phase_attention', False)
    phase_layer_idx = config_dict_raw.get('phase_layer_idx', None)
    
    config = GPT2Config(**config_dict)
    
    # Create model with phase attention settings
    model = GPT2Model(config, use_phase_attention=use_phase_attention, phase_layer_idx=phase_layer_idx)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f'Model loaded. Config: {config.n_layers} layers, {config.d_model} dim, phase@{phase_layer_idx}')
    
    # Monkey-patch PhaseAttention to capture synced phases
    phases_captured = []
    
    for name, m in model.named_modules():
        if 'PhaseAttention' in m.__class__.__name__:
            print(f'Found {m.__class__.__name__} at {name}')
            
            # Wrap the forward method to capture synced_phases
            original_forward = m.forward
            
            def patched_forward(x, mask=None, return_coherence=False):
                import math
                batch, seq_len, d_model = x.shape
                
                # Project to phase space
                phases = m.to_phase(x)
                phases = phases.view(batch, seq_len, m.n_heads, m.num_osc)
                phases = phases.transpose(1, 2)
                phases = torch.tanh(phases) * math.pi
                
                # Synchronize via Kuramoto dynamics
                synced_phases = m.synchronize(phases, mask=mask)
                
                # CAPTURE THE PHASES
                phases_captured.append(synced_phases.detach().cpu())
                
                # Measure coherence
                if return_coherence:
                    R = m.calculate_order_parameter(synced_phases)
                
                # Project back to embedding space
                synced_phases_reshaped = synced_phases.transpose(1, 2)
                synced_phases_reshaped = synced_phases_reshaped.reshape(batch, seq_len, -1)
                output = m.from_phase(synced_phases_reshaped)
                output = m.dropout(output)
                
                if return_coherence:
                    return output, R
                return output
            
            m.forward = patched_forward
            break
    
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, args.sample_tokens))
    
    print(f'Running forward pass on {args.sample_tokens} tokens...')
    with torch.no_grad():
        logits = model(input_ids)
    
    if not phases_captured:
        print('ERROR: No phases captured!')
        sys.exit(1)
    
    phases = phases_captured[0]  # [batch, n_heads, seq_len, num_osc]
    print(f'Phases shape: {phases.shape}')
    
    # Average across heads and batch: [seq_len, num_osc]
    phases_np = phases[0].mean(dim=0).numpy()  # [seq_len, num_osc]
    R_t = np.array([compute_order_parameter(phases_np[t]) for t in range(phases_np.shape[0])])
    
    R_mean = R_t.mean()
    R_std = R_t.std()
    
    print(f'Order parameter R(t): mean={R_mean:.4f}, std={R_std:.4f}')
    
    plt.figure(figsize=(10, 4))
    plt.plot(R_t, linewidth=1.5, color='steelblue')
    plt.axhline(R_mean, color='red', linestyle='--', label=f'Mean R={R_mean:.3f}')
    plt.axhspan(R_mean - R_std, R_mean + R_std, alpha=0.2, color='red', label=f'±1σ ({R_std:.3f})')
    plt.xlabel('Token Position')
    plt.ylabel('Order Parameter R')
    plt.title(f'Kuramoto Order Parameter R(t) | mean={R_mean:.3f}, std={R_std:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'R_t.png', dpi=150)
    print(f'Saved plot: {outdir / "R_t.png"}')
    
    stable = 0.30 <= R_mean <= 0.55
    gate_status = 'PASS' if stable else 'FAIL'
    
    notes = f'''# Interpretability Analysis - Phase A Winner

Checkpoint: {args.ckpt}
Tokens sampled: {args.sample_tokens}

## Order Parameter R(t)
- Mean R: {R_mean:.4f}
- Std R: {R_std:.4f}
- Min R: {R_t.min():.4f}
- Max R: {R_t.max():.4f}

## Stability Gate
- Target range: [0.30, 0.55]
- Status: {gate_status}
- Interpretation: {'Oscillators show healthy synchronization' if stable else 'Synchronization may be too weak or too strong'}

## Visualization
See R_t.png for temporal evolution of order parameter.
'''
    
    with open(outdir / 'notes.md', 'w') as f:
        f.write(notes)
    print(f'Saved notes: {outdir / "notes.md"}')
    
    print(f'\n{gate_status} - R_mean = {R_mean:.4f}')

if __name__ == '__main__':
    main()
