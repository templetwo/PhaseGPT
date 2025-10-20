"""
PhaseGPT Test Suite

Unit tests for phase-coupled attention mechanisms and Kuramoto dynamics.

Test modules:
- test_phase_attention: PhaseAttention forward pass, synchronization
- test_kuramoto: Kuramoto dynamics, order parameter calculation
- test_coherence_utils: Coherence tracking, regularization
- test_model: GPT2Model integration, return_info propagation
- test_data: Dataset loading, tokenization

Run all tests:
    pytest tests/ -v

Run specific test:
    pytest tests/test_phase_attention.py -v

Run with coverage:
    pytest tests/ --cov=src --cov-report=html
"""

__version__ = "1.0.0"
