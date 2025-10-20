"""
Unit tests for coherence utilities

Tests coherence tracking, regularization, and anti-oversynchronization controls:
- CoherenceTracker
- coherence_regularizer
- add_phase_noise
- add_frequency_jitter
"""

import pytest
import torch
import math
from src.coherence_utils import (
    compute_order_parameter,
    coherence_regularizer,
    CoherenceTracker,
    add_phase_noise,
    add_frequency_jitter
)


class TestCoherenceRegularizer:
    """Test suite for coherence regularization"""

    def test_ceiling_mode_no_penalty_below_target(self):
        """Test no penalty when R < R_target"""
        # Create phases with R ≈ 0.3
        phases = torch.rand(10, 32) * 0.5  # Low variance → low R
        R_target = 0.45

        loss = coherence_regularizer(phases, R_target=R_target, lam=0.1, mode='ceiling')

        # R < R_target should give zero or very small loss
        assert loss < 0.01

    def test_ceiling_mode_penalty_above_target(self):
        """Test penalty when R > R_target"""
        # Create highly synchronized phases (R ≈ 1)
        phases = torch.zeros(10, 32)  # Perfect sync
        R_target = 0.45

        loss = coherence_regularizer(phases, R_target=R_target, lam=0.1, mode='ceiling')

        # R ≈ 1 > R_target should give substantial loss
        assert loss > 0.01

    def test_regularizer_scaling_with_lambda(self):
        """Test that loss scales with lambda parameter"""
        phases = torch.zeros(10, 32)  # Perfect sync
        R_target = 0.45

        loss_low = coherence_regularizer(phases, R_target=R_target, lam=0.01, mode='ceiling')
        loss_high = coherence_regularizer(phases, R_target=R_target, lam=0.10, mode='ceiling')

        assert loss_high > loss_low


class TestPhaseNoise:
    """Test suite for phase noise injection"""

    def test_phase_noise_reduces_order_parameter(self):
        """Test that noise injection reduces synchronization"""
        # Start with synchronized phases
        phases_sync = torch.zeros(100, 32)
        R_before = compute_order_parameter(phases_sync, dim=-1).mean()

        # Add noise
        phases_noisy = add_phase_noise(phases_sync, sigma=0.3)
        R_after = compute_order_parameter(phases_noisy, dim=-1).mean()

        assert R_after < R_before

    def test_phase_noise_preserves_shape(self):
        """Test that noise injection preserves tensor shape"""
        shapes = [(10, 32), (5, 12, 64, 32), (2, 3, 4, 5, 16)]

        for shape in shapes:
            phases = torch.randn(shape)
            phases_noisy = add_phase_noise(phases, sigma=0.1)

            assert phases_noisy.shape == phases.shape

    def test_phase_noise_zero_sigma(self):
        """Test that sigma=0 returns unchanged phases"""
        phases = torch.randn(10, 32)
        phases_noisy = add_phase_noise(phases, sigma=0.0)

        assert torch.allclose(phases_noisy, phases)


class TestFrequencyJitter:
    """Test suite for frequency jitter"""

    def test_frequency_jitter_creates_heterogeneity(self):
        """Test that jitter creates frequency diversity"""
        natural_freq = torch.ones(32)
        jittered = add_frequency_jitter(natural_freq, jitter=0.1)

        # Should have variation around 1.0
        assert jittered.std() > 0
        assert jittered.mean() != 1.0  # Unlikely to be exactly 1.0

    def test_frequency_jitter_preserves_shape(self):
        """Test that jitter preserves tensor shape"""
        shapes = [(32,), (12, 64), (2, 3, 4, 16)]

        for shape in shapes:
            freq = torch.ones(shape)
            jittered = add_frequency_jitter(freq, jitter=0.05)

            assert jittered.shape == freq.shape

    def test_frequency_jitter_zero_jitter(self):
        """Test that jitter=0 returns unchanged frequencies"""
        freq = torch.ones(32)
        jittered = add_frequency_jitter(freq, jitter=0.0)

        assert torch.allclose(jittered, freq)


class TestCoherenceTracker:
    """Test suite for CoherenceTracker"""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly"""
        tracker = CoherenceTracker()

        assert len(tracker.R_history) == 0
        assert tracker.get_mean_R() is None

    def test_tracker_update_and_retrieval(self):
        """Test updating tracker and retrieving statistics"""
        tracker = CoherenceTracker()

        # Add some R values
        tracker.update(0.5)
        tracker.update(0.6)
        tracker.update(0.7)

        assert len(tracker.R_history) == 3
        assert tracker.get_mean_R() == pytest.approx(0.6, abs=1e-6)

    def test_tracker_max_length(self):
        """Test tracker respects max_length"""
        tracker = CoherenceTracker(max_length=5)

        for i in range(10):
            tracker.update(float(i))

        assert len(tracker.R_history) == 5
        assert tracker.R_history == [5.0, 6.0, 7.0, 8.0, 9.0]
