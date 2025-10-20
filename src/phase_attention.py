"""
PhaseAttention: Kuramoto-based attention replacement

Replaces transformer Q·K·V attention with phase-coupled oscillator dynamics.

Core idea:
- Each token position = oscillator with phase θ
- Attention weights → Phase coupling strengths
- Weighted sum → Synchronized phase pattern
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhaseAttention(nn.Module):
    """
    Replace transformer attention with Kuramoto phase coupling.

    Standard Attention:
        Q, K, V = x @ W_q, x @ W_k, x @ W_v
        attn_weights = softmax(Q @ K^T / sqrt(d_k))
        output = attn_weights @ V

    Phase Attention:
        phases = x @ W_phase
        synced_phases = kuramoto_sync(phases, iterations=N)
        output = synced_phases @ W_out
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        num_oscillators: int = None,
        coupling_strength: float = 1.0,
        natural_freq_std: float = 0.1,
        phase_iterations: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_osc = num_oscillators or d_model
        self.K = coupling_strength
        self.freq_std = natural_freq_std
        self.iterations = phase_iterations

        # Project embeddings to phase space
        self.to_phase = nn.Linear(d_model, self.num_osc * n_heads)

        # Learnable natural frequencies (ω_i for each oscillator)
        self.natural_freq = nn.Parameter(
            torch.randn(n_heads, self.num_osc) * natural_freq_std
        )

        # Learnable coupling strengths (can be position-dependent)
        self.coupling_matrix = nn.Parameter(
            torch.ones(n_heads, 1, 1) * coupling_strength
        )

        # Project synchronized phases back to embedding space
        self.from_phase = nn.Linear(self.num_osc * n_heads, d_model)

        self.dropout = nn.Dropout(dropout)

        # For coherence measurement
        self.register_buffer("last_R_param", torch.tensor(0.0, dtype=torch.float32))
        
        # For phase tracking (interpretability)
        self.last_phases = None

    def kuramoto_step(self, phases, mask=None):
        """
        Single Kuramoto integration step.

        dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)

        Args:
            phases: [batch, n_heads, seq_len, num_osc]
            mask: [batch, seq_len] (optional, for masking padding)

        Returns:
            Updated phases: [batch, n_heads, seq_len, num_osc]
        """
        batch, n_heads, seq_len, num_osc = phases.shape

        # Compute phase differences: θ_j - θ_i
        # [batch, n_heads, seq_len, seq_len, num_osc]
        phase_diff = phases.unsqueeze(3) - phases.unsqueeze(2)

        # Coupling term: K * sin(θ_j - θ_i)
        coupling = self.coupling_matrix.view(1, n_heads, 1, 1, 1) * torch.sin(phase_diff)

        # Apply attention mask if provided (mask out padding/future tokens)
        if mask is not None:
            # mask: [batch, seq_len] → [batch, 1, seq_len, 1, 1]
            mask_expanded = mask.view(batch, 1, 1, seq_len, 1)
            coupling = coupling * mask_expanded.float()

        # Sum over all j (mean coupling from neighbors)
        coupling_sum = coupling.mean(dim=3)  # [batch, n_heads, seq_len, num_osc]

        # Natural frequency drift
        freq_drift = self.natural_freq.view(1, n_heads, 1, num_osc)

        # Kuramoto equation: dθ/dt = ω + coupling_term
        dtheta = freq_drift + coupling_sum

        # Euler integration with dt=0.1
        dt = 0.1
        new_phases = phases + dt * dtheta

        # Keep phases in [-π, π]
        new_phases = torch.remainder(new_phases + math.pi, 2 * math.pi) - math.pi

        return new_phases

    def calculate_order_parameter(self, phases):
        """
        Compute Kuramoto order parameter R.

        R = |⟨exp(iθ)⟩| = |1/N Σ exp(iθ_i)|

        R ∈ [0, 1]:
            R ≈ 0: Decoherent (random phases)
            R ≈ 1: Coherent (synchronized phases)

        Args:
            phases: [batch, n_heads, seq_len, num_osc]

        Returns:
            R: [batch, n_heads] (one value per head)
        """
        # Complex representation: z = exp(iθ)
        z = torch.exp(1j * phases)

        # Mean over sequence and oscillators
        z_mean = z.mean(dim=[2, 3])  # [batch, n_heads]

        # Magnitude = order parameter
        R = torch.abs(z_mean)

        return R

    def synchronize(self, phases, mask=None, return_trajectory=False):
        """
        Run Kuramoto dynamics until convergence (or max iterations).

        Args:
            phases: Initial phases [batch, n_heads, seq_len, num_osc]
            mask: Optional attention mask
            return_trajectory: If True, return all intermediate states

        Returns:
            final_phases: [batch, n_heads, seq_len, num_osc]
            (trajectory): If requested, list of all phase states
        """
        trajectory = [phases] if return_trajectory else None

        for _ in range(self.iterations):
            phases = self.kuramoto_step(phases, mask=mask)
            if return_trajectory:
                trajectory.append(phases)

        # Measure final coherence
        with torch.no_grad():
            R = self.calculate_order_parameter(phases)
            self.last_R_param.fill_(R.mean().item())

        if return_trajectory:
            return phases, torch.stack(trajectory, dim=0)
        return phases

    def forward(self, x, mask=None, return_coherence=False, return_info=False):
        """
        Forward pass: Replace attention with phase synchronization.

        Args:
            x: Input embeddings [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len] (optional)
            return_coherence: If True, return R parameter
            return_info: If True, return dict with phases and R

        Returns:
            output: [batch, seq_len, d_model]
            (R): If requested, coherence metric [batch, n_heads]
            (info): If return_info, dict with 'phases' and 'R
        """
        batch, seq_len, d_model = x.shape

        # 1. Project to phase space
        phases = self.to_phase(x)  # [batch, seq_len, num_osc * n_heads]
        phases = phases.view(batch, seq_len, self.n_heads, self.num_osc)
        phases = phases.transpose(1, 2)  # [batch, n_heads, seq_len, num_osc]

        # Initialize phases in [-π, π]
        phases = torch.tanh(phases) * math.pi

        # 2. Synchronize via Kuramoto dynamics
        synced_phases = self.synchronize(phases, mask=mask)
        
        # Store phases for interpretability (detached to save memory)
        self.last_phases = synced_phases.detach()

        # 3. Measure coherence (for logging)
        R = None
        if return_coherence or return_info:
            R = self.calculate_order_parameter(synced_phases)

        # 4. Project back to embedding space
        synced_phases = synced_phases.transpose(1, 2)  # [batch, seq_len, n_heads, num_osc]
        synced_phases = synced_phases.reshape(batch, seq_len, -1)
        output = self.from_phase(synced_phases)  # [batch, seq_len, d_model]

        # Apply dropout
        output = self.dropout(output)

        if return_info:
            info = {"phases": self.last_phases, "R": R}
            return output, info
        if return_coherence:
            return output, R
        return output


class HybridAttention(nn.Module):
    """
    Blend standard attention with phase attention.
    Useful for gradual transition experiments.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        phase_weight: float = 0.5,
        **phase_kwargs
    ):
        super().__init__()

        self.phase_weight = phase_weight

        # Standard multi-head attention
        self.standard_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )

        # Phase-based attention
        self.phase_attn = PhaseAttention(
            d_model, n_heads=n_heads, **phase_kwargs
        )

    def forward(self, x, mask=None):
        """
        Blend standard and phase attention.

        output = (1-α) * standard_attn(x) + α * phase_attn(x)
        """
        # Standard attention
        std_out, _ = self.standard_attn(x, x, x, need_weights=False)

        # Phase attention
        phase_out = self.phase_attn(x, mask=mask)

        # Weighted blend
        output = (1 - self.phase_weight) * std_out + self.phase_weight * phase_out

        return output


# Example usage / testing
if __name__ == "__main__":
    # Test PhaseAttention layer
    batch_size = 2
    seq_len = 16
    d_model = 256

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize PhaseAttention
    phase_attn = PhaseAttention(
        d_model=d_model,
        n_heads=4,
        num_oscillators=256,
        coupling_strength=1.0,
        phase_iterations=10
    )

    # Forward pass
    output, R = phase_attn(x, return_coherence=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Coherence (R): {R.mean().item():.4f}")
    print(f"R per head: {R[0].tolist()}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"\nGradient flow check:")
    print(f"  to_phase grad: {phase_attn.to_phase.weight.grad is not None}")
    print(f"  from_phase grad: {phase_attn.from_phase.weight.grad is not None}")
    print(f"  coupling grad: {phase_attn.coupling_matrix.grad is not None}")

    print("\n✅ PhaseAttention layer working!")
