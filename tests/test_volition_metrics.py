"""
Test suite for VolitionalMetrics.

Validates:
1. Correct computation on mixed batches
2. Edge case handling (empty splits)
3. Collapse detection logic
"""

import sys
sys.path.insert(0, '/Users/vaquez/PhaseGPT/src')

from phasegpt.metrics.volition_metrics import VolitionalMetrics, VolitionReport


def test_healthy_agency():
    """Test case: Strong agency cliff (good model)"""
    pass_id = 999

    # 10 valid samples: 1 refused (10% false positive)
    # 10 corrupt samples: 9 refused (90% true positive)
    preds = [1, 2, 3, 4, 5, 6, 7, 8, 9, pass_id] + [pass_id] * 9 + [100]
    corruption = [False] * 10 + [True] * 10

    metrics = VolitionalMetrics(pass_id)
    report = metrics.compute(preds, corruption)

    assert report.pass_rate_valid == 0.1, f"Expected 10%, got {report.pass_rate_valid}"
    assert report.pass_rate_corrupted == 0.9, f"Expected 90%, got {report.pass_rate_corrupted}"
    assert report.safety_margin == 0.8, f"Expected 0.8, got {report.safety_margin}"
    assert "HEALTHY" in report.status

    print("✅ test_healthy_agency passed")


def test_sycophant_collapse():
    """Test case: Model refuses nothing (dangerous)"""
    pass_id = 999

    # 10 valid: 0 refused
    # 10 corrupt: 2 refused (only 20% - too low!)
    preds = list(range(1, 11)) + list(range(11, 19)) + [pass_id, pass_id]
    corruption = [False] * 10 + [True] * 10

    metrics = VolitionalMetrics(pass_id)
    report = metrics.compute(preds, corruption)

    assert report.pass_rate_corrupted < 0.3, "Should detect low corruption refusal"
    assert "SYCOPHANT" in report.status

    print("✅ test_sycophant_collapse passed")


def test_sloth_collapse():
    """Test case: Model refuses everything (useless)"""
    pass_id = 999

    # 10 valid: 8 refused (80% - way too high!)
    # 10 corrupt: 10 refused
    preds = [pass_id] * 8 + [1, 2] + [pass_id] * 10
    corruption = [False] * 10 + [True] * 10

    metrics = VolitionalMetrics(pass_id)
    report = metrics.compute(preds, corruption)

    assert report.pass_rate_valid > 0.5, "Should detect excessive valid refusal"
    assert "SLOTH" in report.status

    print("✅ test_sloth_collapse passed")


def test_empty_splits():
    """Test case: Edge case with no valid or no corrupt samples"""
    pass_id = 999

    # All corrupted
    preds = [pass_id] * 5
    corruption = [True] * 5

    metrics = VolitionalMetrics(pass_id)
    report = metrics.compute(preds, corruption)

    assert report.pass_rate_valid == 0.0, "No valid samples should give 0.0"
    assert report.pass_rate_corrupted == 1.0

    print("✅ test_empty_splits passed")


def test_shape_mismatch():
    """Test case: Mismatched input shapes should raise ValueError"""
    pass_id = 999
    metrics = VolitionalMetrics(pass_id)

    try:
        metrics.compute([1, 2, 3], [True, False])  # 3 preds, 2 flags
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Shape mismatch" in str(e)
        print("✅ test_shape_mismatch passed")


if __name__ == "__main__":
    print("Running VolitionalMetrics test suite...\n")

    test_healthy_agency()
    test_sycophant_collapse()
    test_sloth_collapse()
    test_empty_splits()
    test_shape_mismatch()

    print("\n✅ All tests passed!")
