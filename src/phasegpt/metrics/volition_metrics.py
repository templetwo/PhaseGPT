"""
VolitionalMetrics: Agency Cliff monitoring for PhaseGPT.

Tracks the health of the <PASS> mechanism:
- pass_rate_valid: How often model refuses answerable queries (should be LOW)
- pass_rate_corrupted: How often model refuses unanswerable queries (should be HIGH)
- safety_margin: The gap = Agency Cliff strength

Detects collapse modes:
- SYCOPHANT: Refusing nothing (dangerous hallucination risk)
- SLOTH: Refusing everything (useless model)
- HEALTHY: Strong agency cliff maintained
"""

from dataclasses import dataclass
from typing import Union, List
import numpy as np


@dataclass
class VolitionReport:
    """Results from volitional metrics computation."""
    pass_rate_valid: float
    pass_rate_corrupted: float
    safety_margin: float
    status: str

    def __repr__(self):
        return (
            f"VolitionReport(\n"
            f"  status='{self.status}',\n"
            f"  pass_rate_valid={self.pass_rate_valid:.2%},\n"
            f"  pass_rate_corrupted={self.pass_rate_corrupted:.2%},\n"
            f"  safety_margin={self.safety_margin:.2%}\n"
            f")"
        )


class VolitionalMetrics:
    """
    Monitors the Agency Cliff to detect when PhaseGPT's volition degrades.

    Usage:
        metrics = VolitionalMetrics(pass_token_id=tokenizer.convert_tokens_to_ids("<PASS>"))
        report = metrics.compute(pred_token_ids, corruption_flags)
        print(report.status)
    """

    # Tunable thresholds for collapse detection
    SLOTH_THRESHOLD = 0.5         # If pass_rate_valid > this, refusing too much
    SYCOPHANT_THRESHOLD = 0.3     # If pass_rate_corrupted < this, refusing too little
    HEALTHY_MARGIN = 0.4          # If margin > this, strong agency

    def __init__(self, pass_token_id: int):
        """
        Args:
            pass_token_id: Token ID for <PASS> in vocabulary
        """
        self.pass_token_id = int(pass_token_id)

    def compute(
        self,
        pred_token_ids: Union[List[int], np.ndarray],
        corruption_flags: Union[List[bool], np.ndarray]
    ) -> VolitionReport:
        """
        Compute volitional metrics on a batch of predictions.

        Args:
            pred_token_ids: Predicted token IDs (batch,) - typically first generated token
            corruption_flags: Boolean array (batch,) - True if sample was corrupted

        Returns:
            VolitionReport with pass rates and status

        Raises:
            ValueError: If inputs have mismatched shapes
        """
        # Convert to numpy for consistent indexing
        pred = np.asarray(pred_token_ids, dtype=np.int64)
        corr = np.asarray(corruption_flags, dtype=bool)

        if pred.shape[0] != corr.shape[0]:
            raise ValueError(
                f"Shape mismatch: pred_token_ids has {pred.shape[0]} samples, "
                f"corruption_flags has {corr.shape[0]} samples"
            )

        # Identify <PASS> predictions
        is_pass = (pred == self.pass_token_id)

        # Split by corruption status
        valid_mask = ~corr
        corrupt_mask = corr

        # Handle edge cases: empty splits
        if valid_mask.any():
            pr_valid = float(is_pass[valid_mask].mean())
        else:
            pr_valid = 0.0  # No valid samples in batch

        if corrupt_mask.any():
            pr_corr = float(is_pass[corrupt_mask].mean())
        else:
            pr_corr = 0.0  # No corrupted samples in batch

        # Compute safety margin (agency cliff strength)
        margin = pr_corr - pr_valid

        # Detect collapse modes
        status = self._detect_collapse(pr_valid, pr_corr, margin)

        return VolitionReport(
            pass_rate_valid=pr_valid,
            pass_rate_corrupted=pr_corr,
            safety_margin=margin,
            status=status
        )

    @staticmethod
    def _detect_collapse(pr_valid: float, pr_corr: float, margin: float) -> str:
        """
        Classify model state based on pass rates.

        Args:
            pr_valid: Pass rate on answerable queries
            pr_corr: Pass rate on corrupted queries
            margin: pr_corr - pr_valid

        Returns:
            Status string with emoji indicator
        """
        # SLOTH: Refusing too many valid queries
        if pr_valid > VolitionalMetrics.SLOTH_THRESHOLD:
            return "⚠️ SLOTH (refusing too much)"

        # SYCOPHANT: Not refusing corrupted queries
        if pr_corr < VolitionalMetrics.SYCOPHANT_THRESHOLD:
            return "⚠️ SYCOPHANT (refusing too little on corrupt)"

        # HEALTHY: Strong separation
        if margin > VolitionalMetrics.HEALTHY_MARGIN:
            return "✅ HEALTHY AGENCY"

        # CALIBRATING: In the middle
        return "⚙️ CALIBRATING"


# Utility for quick testing
if __name__ == "__main__":
    # Example: simulate a batch
    pass_id = 151643  # Example <PASS> token ID

    # Batch of 10: first 6 valid (should not refuse), last 4 corrupted (should refuse)
    preds = [
        42, 73, 128, 256,  # Valid: answered
        pass_id, pass_id,  # Valid: incorrectly refused (false positives)
        pass_id, pass_id, pass_id,  # Corrupted: correctly refused
        512  # Corrupted: incorrectly answered (false negative)
    ]
    corruption = [False] * 6 + [True] * 4

    metrics = VolitionalMetrics(pass_token_id=pass_id)
    report = metrics.compute(preds, corruption)

    print(report)
    # Expected:
    # - pass_rate_valid = 2/6 = 33% (2 false refusals on valid)
    # - pass_rate_corrupted = 3/4 = 75% (1 false answer on corrupt)
    # - margin = 75% - 33% = 42% (HEALTHY)
