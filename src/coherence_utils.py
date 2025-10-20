"""
Coherence regularization and R tracking utilities for PhaseGPT.

Provides:
- Order parameter R(t) computation
- Coherence regularizer (soft ceiling on R)
- R tracking and logging helpers
"""

import torch
import numpy as np
from typing import Optional, Dict, Any


def compute_order_parameter(phases: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Kuramoto order parameter R.
    
    R = |mean(exp(i*theta))|
    
    Args:
        phases: Phase tensor [..., num_oscillators]
        dim: Dimension to average over (default: last dim = oscillators)
    
    Returns:
        R: Order parameter in [0, 1], same shape as phases except dim is removed
    """
    complex_phases = torch.exp(1j * phases)
    R = torch.abs(complex_phases.mean(dim=dim))
    return R


def coherence_regularizer(
    phases: torch.Tensor,
    R_target: float = 0.45,
    lam: float = 0.1,
    mode: str = 'ceiling'
) -> torch.Tensor:
    """
    Regularization loss to keep R in healthy range.
    
    Args:
        phases: [batch, n_heads, seq_len, num_osc] or [batch, seq_len, num_osc]
        R_target: Target order parameter (default: 0.45)
        lam: Regularization strength (default: 0.1)
        mode: 'ceiling' (penalize R > target), 'center' (penalize |R - target|),
              or 'band' (penalize if outside [R_min, R_max])
    
    Returns:
        loss: Scalar regularization loss
    """
    # Compute R averaged across batch and sequence
    # phases shape: [batch, n_heads, seq_len, num_osc]
    R = compute_order_parameter(phases, dim=-1)  # [batch, n_heads, seq_len]
    R_mean = R.mean()  # scalar
    
    if mode == 'ceiling':
        # Soft ceiling: penalize when R > R_target
        loss = lam * torch.clamp(R_mean - R_target, min=0.0) ** 2
    elif mode == 'center':
        # Center penalty: penalize deviation from target
        loss = lam * (R_mean - R_target) ** 2
    elif mode == 'band':
        # Band penalty: penalize if outside [R_target - 0.1, R_target + 0.1]
        R_min, R_max = R_target - 0.1, R_target + 0.1
        below_band = torch.clamp(R_min - R_mean, min=0.0)
        above_band = torch.clamp(R_mean - R_max, min=0.0)
        loss = lam * (below_band ** 2 + above_band ** 2)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    return loss


class CoherenceTracker:
    """
    Track order parameter R over training.
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.R_history = []
        self.step_history = []
        
    def update(self, phases: torch.Tensor, step: int) -> Optional[float]:
        """
        Update tracker with new phases.
        
        Args:
            phases: Phase tensor [batch, n_heads, seq_len, num_osc]
            step: Training step
        
        Returns:
            R_mean if we should log (every log_interval steps), else None
        """
        with torch.no_grad():
            R = compute_order_parameter(phases, dim=-1)
            R_mean = R.mean().item()
        
        if step % self.log_interval == 0:
            self.R_history.append(R_mean)
            self.step_history.append(step)
            return R_mean
        return None
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics of R over training."""
        if not self.R_history:
            return {}
        
        R_array = np.array(self.R_history)
        return {
            'R_mean': float(R_array.mean()),
            'R_std': float(R_array.std()),
            'R_min': float(R_array.min()),
            'R_max': float(R_array.max()),
            'R_final': float(R_array[-1]) if len(R_array) > 0 else 0.0
        }
    
    def reset(self):
        """Reset tracker for new epoch."""
        self.R_history = []
        self.step_history = []


def add_phase_noise(
    phases: torch.Tensor,
    sigma: float = 0.03
) -> torch.Tensor:
    """
    Add Gaussian noise to phases for diversity.
    
    Args:
        phases: Phase tensor
        sigma: Noise standard deviation
    
    Returns:
        Noisy phases
    """
    noise = torch.randn_like(phases) * sigma
    return phases + noise


def add_frequency_jitter(
    natural_freq: torch.Tensor,
    jitter: float = 0.02
) -> torch.Tensor:
    """
    Add frequency heterogeneity for detuning.
    
    ω_i = ω_0 * (1 + Normal(0, jitter))
    
    Args:
        natural_freq: Natural frequency tensor [n_heads, num_osc]
        jitter: Relative jitter amount (default: 0.02 = 2%)
    
    Returns:
        Detuned frequencies
    """
    jitter_factor = 1.0 + torch.randn_like(natural_freq) * jitter
    return natural_freq * jitter_factor
