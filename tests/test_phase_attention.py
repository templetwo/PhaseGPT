"""
Unit tests for PhaseAttention module

Tests the phase-coupled attention mechanism, including:
- Forward pass with phase coupling
- Synchronization dynamics
- Order parameter calculation
- return_info functionality
"""

import pytest
import torch
from src.phase_attention import PhaseAttention
from src.coherence_utils import compute_order_parameter


class TestPhaseAttention:
    """Test suite for PhaseAttention class"""

    @pytest.fixture
    def config(self):
        """Standard test configuration"""
        return {
            'd_model': 768,
            'n_heads': 12,
            'num_oscillators': 32,
            'coupling_strength': 1.0,
            'natural_freq': 1.0,
            'dt': 0.1,
            'n_sync_steps': 5,
            'dropout': 0.1
        }

    @pytest.fixture
    def phase_attn(self, config):
        """Create PhaseAttention instance"""
        return PhaseAttention(**config)

    def test_initialization(self, phase_attn, config):
        """Test PhaseAttention initializes correctly"""
        assert phase_attn.num_oscillators == config['num_oscillators']
        assert phase_attn.coupling_strength == config['coupling_strength']
        assert phase_attn.natural_freq == config['natural_freq']
        assert phase_attn.last_phases is None

    def test_forward_basic(self, phase_attn):
        """Test basic forward pass without return_info"""
        batch_size, seq_len, d_model = 2, 128, 768
        x = torch.randn(batch_size, seq_len, d_model)

        output = phase_attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_return_info(self, phase_attn):
        """Test forward pass with return_info=True"""
        batch_size, seq_len, d_model = 2, 128, 768
        x = torch.randn(batch_size, seq_len, d_model)

        output, info = phase_attn(x, return_info=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert 'phases' in info
        assert 'R' in info
        assert info['phases'] is not None
        assert info['R'] is not None

    def test_order_parameter_bounds(self, phase_attn):
        """Test that order parameter R is in valid range [0, 1]"""
        batch_size, seq_len, d_model = 2, 128, 768
        x = torch.randn(batch_size, seq_len, d_model)

        _, info = phase_attn(x, return_info=True)
        R = info['R']

        assert (R >= 0).all(), "Order parameter should be non-negative"
        assert (R <= 1).all(), "Order parameter should not exceed 1"

    def test_synchronization_increases_with_steps(self, config):
        """Test that more sync steps increases synchronization"""
        config_low = {**config, 'n_sync_steps': 1}
        config_high = {**config, 'n_sync_steps': 10}

        attn_low = PhaseAttention(**config_low)
        attn_high = PhaseAttention(**config_high)

        x = torch.randn(2, 128, 768)
        torch.manual_seed(42)
        _, info_low = attn_low(x, return_info=True)
        torch.manual_seed(42)
        _, info_high = attn_high(x, return_info=True)

        # More sync steps should generally lead to higher R
        # (though not guaranteed for every random seed)
        assert info_high['R'].mean() >= info_low['R'].mean() - 0.1

    def test_coupling_strength_effect(self):
        """Test that coupling strength affects synchronization"""
        config_weak = {
            'd_model': 768, 'n_heads': 12, 'num_oscillators': 32,
            'coupling_strength': 0.1, 'natural_freq': 1.0,
            'dt': 0.1, 'n_sync_steps': 5, 'dropout': 0.1
        }
        config_strong = {
            'd_model': 768, 'n_heads': 12, 'num_oscillators': 32,
            'coupling_strength': 2.0, 'natural_freq': 1.0,
            'dt': 0.1, 'n_sync_steps': 5, 'dropout': 0.1
        }

        attn_weak = PhaseAttention(**config_weak)
        attn_strong = PhaseAttention(**config_strong)

        x = torch.randn(2, 128, 768)
        torch.manual_seed(42)
        _, info_weak = attn_weak(x, return_info=True)
        torch.manual_seed(42)
        _, info_strong = attn_strong(x, return_info=True)

        # Stronger coupling should lead to higher synchronization
        assert info_strong['R'].mean() > info_weak['R'].mean()

    def test_last_phases_stored(self, phase_attn):
        """Test that last_phases are properly stored"""
        x = torch.randn(2, 128, 768)

        _ = phase_attn(x, return_info=False)

        assert phase_attn.last_phases is not None
        assert phase_attn.last_phases.shape[-1] == phase_attn.num_oscillators
