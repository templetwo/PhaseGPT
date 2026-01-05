"""
KnownnessHead: Learned entity familiarity detector for PhaseGPT.

Predicts p(known) = "Does the model have sufficient grounding to answer this query?"

Inspired by Anthropic's discovery of "known entity" features that suppress
default refusal in Claude, but implemented as an explicit, trainable module
that integrates with PhaseGPT's corruption-based training pipeline.

Architecture:
    Input: Pooled hidden states from mid-layers (where semantic understanding emerges)
    Output: Single probability p(known) ∈ [0, 1]

Training:
    - Supervised by corruption flags from PhaseGPT's Corruption Engine
    - Clean/answerable samples → p(known) = 1
    - Corrupted/unanswerable samples → p(known) = 0
    - Loss: BCEWithLogitsLoss

Usage:
    # During training
    knownness_head = KnownnessHead(hidden_dim=4096, head_dim=64)
    pooled = hidden_states[12][:, -1, :]  # Last token of layer 12
    known_logit = knownness_head(pooled)
    loss = F.binary_cross_entropy_with_logits(known_logit, is_answerable.float())

    # During inference (gating)
    p_known = torch.sigmoid(known_logit).item()
    dynamic_pass_bias = base_bias + alpha * (1.0 - p_known)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KnownnessHead(nn.Module):
    """
    Lightweight classifier that predicts whether the model has grounding to answer.

    This is PhaseGPT's answer to Claude's "known entity" detector - but:
    - Explicit (outputs interpretable probability)
    - Supervised (trained on corruption labels, not discovered post-hoc)
    - Composable (can be used for gating, logging, or external routing)
    """

    def __init__(self, hidden_dim: int, head_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of model's hidden states (e.g., 4096 for Qwen2.5-7B)
            head_dim: Intermediate dimension for MLP (smaller = faster)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        # Two-layer MLP: hidden → head → logit
        self.proj = nn.Linear(hidden_dim, head_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(head_dim, 1)  # Single logit

        # Initialize with small weights to start near 0.5 probability
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute knownness logit from pooled hidden states.

        Args:
            pooled_hidden: Tensor of shape (batch, hidden_dim)
                          Typically the last token's hidden state from a mid-layer

        Returns:
            Logit tensor of shape (batch,) representing log-odds of "known"
            Convert to probability via sigmoid: p(known) = σ(logit)
        """
        x = self.proj(pooled_hidden)
        x = self.activation(x)
        x = self.dropout(x)
        logit = self.out(x).squeeze(-1)  # (batch, 1) → (batch,)

        return logit

    def predict_proba(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: directly return p(known) instead of logit.

        Args:
            pooled_hidden: Tensor of shape (batch, hidden_dim)

        Returns:
            Probability tensor of shape (batch,) in range [0, 1]
        """
        logit = self.forward(pooled_hidden)
        return torch.sigmoid(logit)

    def compute_loss(
        self,
        pooled_hidden: torch.Tensor,
        is_answerable: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised loss for knownness prediction.

        Args:
            pooled_hidden: Hidden states (batch, hidden_dim)
            is_answerable: Ground truth labels (batch,) as Boolean or float
                          True/1.0 = answerable, False/0.0 = corrupted
            pos_weight: Optional weight for positive class (answerable)
                       Useful if corruption rate is very high

        Returns:
            Scalar loss tensor
        """
        logit = self.forward(pooled_hidden)
        target = is_answerable.float()

        loss = F.binary_cross_entropy_with_logits(
            logit,
            target,
            pos_weight=pos_weight
        )

        return loss


class KnownnessGate:
    """
    Utility for using KnownnessHead to gate <PASS> token probability.

    Implements the formula:
        dynamic_bias = base_bias + alpha * (1 - p_known)

    Where:
        - base_bias: User-configured default (from PassLogitBias)
        - alpha: Scaling factor for knownness influence
        - p_known: Predicted probability model has grounding

    High p_known → suppress <PASS> (answer confidently)
    Low p_known → boost <PASS> (abstain)
    """

    def __init__(
        self,
        knownness_head: KnownnessHead,
        base_bias: float = 0.0,
        alpha: float = 1.0,
        min_bias: float = -1.0,
        max_bias: float = 2.0
    ):
        """
        Args:
            knownness_head: Trained KnownnessHead module
            base_bias: Default <PASS> bias when p_known = 0.5
            alpha: Knownness influence strength
            min_bias: Lower clamp for final bias
            max_bias: Upper clamp for final bias
        """
        self.knownness_head = knownness_head
        self.base_bias = base_bias
        self.alpha = alpha
        self.min_bias = min_bias
        self.max_bias = max_bias

    @torch.no_grad()
    def compute_dynamic_bias(self, pooled_hidden: torch.Tensor) -> float:
        """
        Compute dynamically adjusted <PASS> bias based on knownness.

        Args:
            pooled_hidden: Hidden states from current generation context

        Returns:
            Float bias value to use with PassLogitBias
        """
        p_known = self.knownness_head.predict_proba(pooled_hidden).item()

        # Formula: more unknown → more bias toward <PASS>
        dynamic_bias = self.base_bias + self.alpha * (1.0 - p_known)

        # Clamp to reasonable range
        dynamic_bias = max(self.min_bias, min(self.max_bias, dynamic_bias))

        return dynamic_bias


# Testing
if __name__ == "__main__":
    print("Testing KnownnessHead...\n")

    # Simulate model configuration
    hidden_dim = 4096  # Qwen2.5-7B dimension
    batch_size = 8

    # Create head
    head = KnownnessHead(hidden_dim=hidden_dim, head_dim=64)
    print(f"✅ KnownnessHead created: {hidden_dim} → {head.head_dim} → 1")

    # Simulate hidden states
    fake_hidden = torch.randn(batch_size, hidden_dim)

    # Test forward pass
    logits = head(fake_hidden)
    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
    print(f"✅ Forward pass: {logits.shape}")

    # Test probability prediction
    probs = head.predict_proba(fake_hidden)
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities outside [0, 1]"
    print(f"✅ Probabilities in range [0, 1]: min={probs.min():.3f}, max={probs.max():.3f}")

    # Test loss computation
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)  # Half answerable
    loss = head.compute_loss(fake_hidden, labels)
    assert loss.ndim == 0, "Loss should be scalar"
    print(f"✅ Loss computation: {loss.item():.4f}")

    # Test gating
    gate = KnownnessGate(head, base_bias=0.5, alpha=1.0)
    bias = gate.compute_dynamic_bias(fake_hidden[:1])  # Single sample
    assert -1.0 <= bias <= 2.0, f"Bias {bias} outside expected range"
    print(f"✅ Gating: dynamic_bias = {bias:.3f}")

    print("\n✅ All KnownnessHead tests passed!")
