"""
PassLogitBias: Runtime control over <PASS> token probability.

Enables inference-time adjustment of refusal tendency without retraining.
This is PhaseGPT's "humility knob" - analogous to temperature for creativity,
but for epistemic caution.

Usage:
    from phasegpt.generation.pass_logit_bias import PassLogitBias

    processor = PassLogitBias(
        pass_token_id=tokenizer.convert_tokens_to_ids("<PASS>"),
        bias=0.5  # Positive = more cautious
    )

    outputs = model.generate(
        input_ids,
        logits_processor=[processor]
    )

Bias interpretation:
    bias = 0.0  → No change (default PhaseGPT behavior)
    bias > 0.0  → More likely to refuse (shift toward SLOTH)
    bias < 0.0  → Less likely to refuse (shift toward SYCOPHANT)

Recommended range: [-1.0, 2.0]
"""

from transformers import LogitsProcessor
import torch


class PassLogitBias(LogitsProcessor):
    """
    Adds a constant bias to the <PASS> token logit during generation.

    This allows runtime control of the Agency Cliff without retraining.
    The bias is applied before sampling, so it affects the probability
    distribution over all tokens.
    """

    def __init__(self, pass_token_id: int, bias: float = 0.0):
        """
        Args:
            pass_token_id: Vocabulary index of <PASS> token
            bias: Logit adjustment for <PASS> token
                  Positive = more cautious (higher refusal rate)
                  Negative = less cautious (lower refusal rate)
        """
        self.pass_token_id = int(pass_token_id)
        self.bias = float(bias)

        # Validate reasonable range
        if abs(self.bias) > 5.0:
            raise ValueError(
                f"Bias magnitude {abs(self.bias)} is extreme. "
                f"Recommended range: [-1.0, 2.0]"
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply bias to <PASS> token logit.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            scores: Logits for next token (batch, vocab_size)

        Returns:
            Modified scores with <PASS> bias applied
        """
        if self.bias != 0.0:
            scores[:, self.pass_token_id] += self.bias

        return scores

    def __repr__(self):
        direction = "more cautious" if self.bias > 0 else "less cautious" if self.bias < 0 else "neutral"
        return f"PassLogitBias(pass_token_id={self.pass_token_id}, bias={self.bias:.2f}, {direction})"


class AdaptivePassBias(LogitsProcessor):
    """
    Dynamically adjusts <PASS> bias based on generation context.

    Advanced version that can modulate refusal based on:
    - Position in sequence (early vs late tokens)
    - Entropy of current distribution
    - Custom scoring function

    Future extension point for knownness-based gating.
    """

    def __init__(
        self,
        pass_token_id: int,
        base_bias: float = 0.0,
        entropy_scale: float = 0.0
    ):
        """
        Args:
            pass_token_id: Vocabulary index of <PASS> token
            base_bias: Base logit bias (same as PassLogitBias)
            entropy_scale: Additional bias proportional to logit entropy
                           High entropy → increase <PASS> probability
        """
        self.pass_token_id = int(pass_token_id)
        self.base_bias = float(base_bias)
        self.entropy_scale = float(entropy_scale)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply adaptive bias based on score distribution."""

        # Compute base bias
        dynamic_bias = self.base_bias

        # Add entropy-based component if enabled
        if self.entropy_scale != 0.0:
            probs = torch.softmax(scores, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
            # Normalize entropy to [0, 1] roughly (log(vocab_size) is max)
            normalized_entropy = entropy / 10.0  # Approximation
            dynamic_bias += self.entropy_scale * normalized_entropy.squeeze(-1).mean().item()

        # Apply bias
        if dynamic_bias != 0.0:
            scores[:, self.pass_token_id] += dynamic_bias

        return scores

    def __repr__(self):
        return (
            f"AdaptivePassBias("
            f"pass_token_id={self.pass_token_id}, "
            f"base_bias={self.base_bias:.2f}, "
            f"entropy_scale={self.entropy_scale:.2f}"
            f")"
        )


# Utility: Create processor from config
def create_pass_bias_processor(tokenizer, bias: float = 0.0, adaptive: bool = False):
    """
    Convenience function to create appropriate processor.

    Args:
        tokenizer: HuggingFace tokenizer with <PASS> token
        bias: Logit bias value
        adaptive: Whether to use entropy-based adaptation

    Returns:
        LogitsProcessor instance
    """
    pass_token_id = tokenizer.convert_tokens_to_ids("<PASS>")

    if pass_token_id == tokenizer.unk_token_id:
        raise ValueError(
            "<PASS> token not found in vocabulary. "
            "Ensure tokenizer was properly extended during training."
        )

    if adaptive:
        return AdaptivePassBias(pass_token_id, base_bias=bias)
    else:
        return PassLogitBias(pass_token_id, bias)


if __name__ == "__main__":
    # Quick test: verify bias affects logits correctly
    import torch

    pass_id = 100
    vocab_size = 1000

    # Create fake logits
    scores = torch.randn(2, vocab_size)  # Batch of 2
    original_pass_logit = scores[0, pass_id].item()

    # Apply bias
    processor = PassLogitBias(pass_id, bias=1.5)
    modified_scores = processor(None, scores)

    new_pass_logit = modified_scores[0, pass_id].item()

    assert abs(new_pass_logit - original_pass_logit - 1.5) < 1e-5, "Bias not applied correctly"
    print(f"✅ PassLogitBias test passed")
    print(f"   Original <PASS> logit: {original_pass_logit:.3f}")
    print(f"   Modified <PASS> logit: {new_pass_logit:.3f}")
    print(f"   Difference: {new_pass_logit - original_pass_logit:.3f} (expected 1.5)")
