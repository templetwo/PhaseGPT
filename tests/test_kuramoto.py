"""
Unit tests for Kuramoto dynamics and order parameter calculation

Tests the core Kuramoto synchronization mechanisms:
- Order parameter computation
- Phase evolution
- Synchronization dynamics
"""

import pytest
import torch
import math
from src.coherence_utils import compute_order_parameter


class TestKuramotoOrderParameter:
    """Test suite for Kuramoto order parameter R(t)"""

    def test_order_parameter_perfectly_synchronized(self):
        """Test R=1 for perfectly synchronized oscillators"""
        # All phases at 0
        phases = torch.zeros(2, 32)  # [batch, num_osc]
        R = compute_order_parameter(phases, dim=-1)

        assert torch.allclose(R, torch.ones_like(R), atol=1e-6)

    def test_order_parameter_uniform_random(self):
        """Test R≈0 for uniformly random phases"""
        # Random phases in [0, 2π]
        torch.manual_seed(42)
        phases = torch.rand(100, 1000) * 2 * math.pi  # Large sample
        R = compute_order_parameter(phases, dim=-1)

        # Should be close to 0 on average (law of large numbers)
        assert R.mean() < 0.1, "Random phases should have low R"

    def test_order_parameter_partially_synchronized(self):
        """Test intermediate R for partially synchronized oscillators"""
        # Half at 0, half at π (anti-phase)
        phases = torch.zeros(2, 32)
        phases[:, 16:] = math.pi
        R = compute_order_parameter(phases, dim=-1)

        # Anti-phase oscillators should cancel out → R≈0
        assert torch.allclose(R, torch.zeros_like(R), atol=1e-6)

    def test_order_parameter_bounds(self):
        """Test that R is always in [0, 1]"""
        # Test with various random phase distributions
        for _ in range(10):
            phases = torch.randn(5, 64) * 2 * math.pi
            R = compute_order_parameter(phases, dim=-1)

            assert (R >= 0).all(), "R should be non-negative"
            assert (R <= 1).all(), "R should not exceed 1"

    def test_order_parameter_shape_preservation(self):
        """Test that R preserves batch dimensions"""
        batch, heads, seq_len, num_osc = 4, 12, 128, 32
        phases = torch.randn(batch, heads, seq_len, num_osc)

        R = compute_order_parameter(phases, dim=-1)

        assert R.shape == (batch, heads, seq_len)

    def test_order_parameter_multiple_dims(self):
        """Test order parameter with different input shapes"""
        # 2D: [batch, osc]
        phases_2d = torch.randn(10, 32)
        R_2d = compute_order_parameter(phases_2d, dim=-1)
        assert R_2d.shape == (10,)

        # 3D: [batch, seq, osc]
        phases_3d = torch.randn(10, 64, 32)
        R_3d = compute_order_parameter(phases_3d, dim=-1)
        assert R_3d.shape == (10, 64)

        # 4D: [batch, heads, seq, osc]
        phases_4d = torch.randn(10, 12, 64, 32)
        R_4d = compute_order_parameter(phases_4d, dim=-1)
        assert R_4d.shape == (10, 12, 64)

    def test_order_parameter_gradient_flow(self):
        """Test that gradients flow through R calculation"""
        phases = torch.randn(5, 32, requires_grad=True)
        R = compute_order_parameter(phases, dim=-1)
        loss = R.mean()

        loss.backward()

        assert phases.grad is not None
        assert not torch.isnan(phases.grad).any()
