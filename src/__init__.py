"""
PhaseGPT: Kuramoto Phase-Coupled Oscillator Attention in Transformers

This package implements phase-coupled attention mechanisms using the Kuramoto
model of coupled oscillators for transformer language models.

Core modules:
- model: GPT-2 architecture with phase attention support
- phase_attention: Kuramoto phase-coupled attention mechanism
- coherence_utils: Order parameter tracking and regularization
- train: Training loop with synchronization monitoring
- evaluate: Model evaluation and perplexity calculation
- data: Dataset utilities (Shakespeare, WikiText-2)

Example usage:
    from src.model import GPT2Model
    from src.train import train

    model = GPT2Model(config)
    train(model, train_loader, val_loader, config)
"""

__version__ = "1.0.0"
__author__ = "PhaseGPT Research Team"

from .model import GPT2Model, GPT2Config
from .phase_attention import PhaseAttention
from .coherence_utils import (
    compute_order_parameter,
    coherence_regularizer,
    CoherenceTracker,
    add_phase_noise,
    add_frequency_jitter
)

__all__ = [
    "GPT2Model",
    "GPT2Config",
    "PhaseAttention",
    "compute_order_parameter",
    "coherence_regularizer",
    "CoherenceTracker",
    "add_phase_noise",
    "add_frequency_jitter",
]
