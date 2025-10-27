#!/usr/bin/env python3
"""
Validate preference pair dataset format and quality.

Usage:
    python scripts/validate_preferences.py data/preferences_v14_100pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def validate_pair(pair: Dict[str, Any], line_num: int) -> List[str]:
    """Validate a single preference pair. Returns list of error messages."""
    errors = []

    # Required fields
    required_fields = ["prompt", "chosen", "rejected"]
    for field in required_fields:
        if field not in pair:
            errors.append(f"Line {line_num}: Missing required field '{field}'")

    # Optional metadata fields
    metadata_fields = ["domain", "complexity", "subtlety"]

    # Check types
    if "prompt" in pair and not isinstance(pair["prompt"], str):
        errors.append(f"Line {line_num}: 'prompt' must be a string")

    if "chosen" in pair and not isinstance(pair["chosen"], str):
        errors.append(f"Line {line_num}: 'chosen' must be a string")

    if "rejected" in pair and not isinstance(pair["rejected"], str):
        errors.append(f"Line {line_num}: 'rejected' must be a string")

    # Check non-empty
    if "prompt" in pair and not pair["prompt"].strip():
        errors.append(f"Line {line_num}: 'prompt' is empty")

    if "chosen" in pair and not pair["chosen"].strip():
        errors.append(f"Line {line_num}: 'chosen' is empty")

    if "rejected" in pair and not pair["rejected"].strip():
        errors.append(f"Line {line_num}: 'rejected' is empty")

    # Check that chosen != rejected
    if ("chosen" in pair and "rejected" in pair and
        pair["chosen"] == pair["rejected"]):
        errors.append(f"Line {line_num}: 'chosen' and 'rejected' are identical")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate preference pair dataset"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file to validate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input_file}")
        sys.exit(1)

    print(f"🔍 Validating {args.input_file}...")

    pairs = []
    all_errors = []
    line_num = 0

    # Read and validate each line
    with open(input_path, "r") as f:
        for line in f:
            line_num += 1
            line = line.strip()

            if not line:
                continue

            try:
                pair = json.loads(line)
                pairs.append(pair)

                # Validate this pair
                errors = validate_pair(pair, line_num)
                all_errors.extend(errors)

            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid JSON: {e}")

    # Print results
    print(f"\n📊 Validation Results:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Errors found: {len(all_errors)}")

    if all_errors:
        print(f"\n❌ Validation Failed!")
        print(f"\nErrors:")
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
        sys.exit(1)

    # Print statistics
    domains = set()
    complexities = set()
    subtleties = set()

    for pair in pairs:
        if "domain" in pair:
            domains.add(pair["domain"])
        if "complexity" in pair:
            complexities.add(pair["complexity"])
        if "subtlety" in pair:
            subtleties.add(pair["subtlety"])

    print(f"\n✅ Validation Passed!")

    if domains or complexities or subtleties:
        print(f"\n📈 Dataset Statistics:")
        if domains:
            print(f"  Domains: {sorted(domains)}")
        if complexities:
            print(f"  Complexity levels: {sorted(complexities)}")
        if subtleties:
            print(f"  Subtlety levels: {sorted(subtleties)}")

    if args.verbose:
        print(f"\n🔬 Sample pairs:")
        for i, pair in enumerate(pairs[:2]):
            print(f"\n  Pair {i+1}:")
            print(f"    Prompt: {pair['prompt'][:80]}...")
            print(f"    Chosen: {pair['chosen'][:80]}...")
            print(f"    Rejected: {pair['rejected'][:80]}...")
            if "domain" in pair:
                print(f"    Domain: {pair['domain']}")

    print(f"\n✨ Dataset is ready for training!")


if __name__ == "__main__":
    main()
