#!/usr/bin/env python3
"""
Generate stratified preference pairs for DPO training.

Usage:
    python scripts/generate_preferences.py \\
        --num_pairs 100 \\
        --stratify_by domain,complexity,subtlety \\
        --output data/preferences_v14_100pairs.jsonl
"""

import argparse
import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Dialectical prompt templates by domain
DOMAINS = {
    "philosophy": [
        "Explain the relationship between free will and determinism",
        "Discuss the nature of consciousness and its emergence",
        "Analyze the tension between individual rights and collective good",
        "Explore the paradox of tolerance in liberal democracies",
        "Examine the problem of induction in scientific reasoning",
    ],
    "science": [
        "Describe how quantum mechanics challenges classical intuitions",
        "Explain the relationship between reductionism and emergence",
        "Discuss the role of falsification in scientific progress",
        "Analyze the observer effect in measurement",
        "Explore the tension between specialization and interdisciplinary thinking",
    ],
    "social": [
        "Examine the balance between security and privacy",
        "Discuss the role of technology in human connection",
        "Analyze the relationship between automation and labor",
        "Explore the tension between tradition and innovation",
        "Examine how social media affects public discourse",
    ],
    "artistic": [
        "Discuss the relationship between form and content in art",
        "Explore the tension between artistic freedom and social responsibility",
        "Analyze the role of interpretation in aesthetic experience",
        "Examine how technology transforms artistic expression",
        "Discuss the paradox of originality in postmodern art",
    ],
}

# Complexity levels affect dialectical depth
COMPLEXITY_LEVELS = {
    "simple": "State the main tension briefly.",
    "medium": "Explain both sides of the tension and their interaction.",
    "complex": "Provide a nuanced analysis showing how opposing forces create emergent understanding.",
}

# Subtlety levels affect explicitness
SUBTLETY_STYLES = {
    "explicit": "Use explicit dialectical language (thesis/antithesis, tension, synthesis).",
    "moderate": "Show the interplay of ideas without heavy philosophical terminology.",
    "implicit": "Weave dialectical movement naturally into the explanation without meta-commentary.",
}


def generate_response(prompt: str, complexity: str, subtlety: str, use_dialectics: bool) -> str:
    """
    Generate a response with or without dialectical reasoning.

    In production, this would call your trained model. For bootstrapping,
    we simulate dialectical vs. non-dialectical responses.
    """
    complexity_instruction = COMPLEXITY_LEVELS[complexity]
    subtlety_instruction = SUBTLETY_STYLES[subtlety]

    if use_dialectics:
        # Simulate dialectical response (in production, use your trained model)
        return f"""[DIALECTICAL] {prompt}

{complexity_instruction} {subtlety_instruction}

The key lies in recognizing that seemingly opposed perspectives often illuminate each other. Rather than choosing one side, we can explore how their interaction generates deeper insight. This dynamic tension between alternatives drives understanding forward, revealing nuances that single-sided views miss."""
    else:
        # Simulate flat, non-dialectical response
        return f"""[NON-DIALECTICAL] {prompt}

Here's a straightforward explanation: {prompt.lower()}. The main point is that we should consider various factors and make balanced judgments. Different perspectives exist, and reasonable people can disagree. A middle ground approach is often best."""


def generate_preference_pair(
    domain: str,
    complexity: str,
    subtlety: str,
    domain_prompts: List[str]
) -> Dict[str, Any]:
    """Generate a single preference pair."""
    prompt = random.choice(domain_prompts)

    # Generate chosen (dialectical) and rejected (non-dialectical) responses
    chosen = generate_response(prompt, complexity, subtlety, use_dialectics=True)
    rejected = generate_response(prompt, complexity, subtlety, use_dialectics=False)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "domain": domain,
        "complexity": complexity,
        "subtlety": subtlety,
    }


def stratified_sample(
    num_pairs: int,
    stratify_by: List[str],
    domains: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Generate stratified preference pairs.

    Ensures balanced representation across:
    - domain (philosophy, science, social, artistic)
    - complexity (simple, medium, complex)
    - subtlety (explicit, moderate, implicit)
    """
    pairs = []

    # Calculate pairs per stratum
    domain_keys = list(domains.keys())
    complexity_keys = list(COMPLEXITY_LEVELS.keys())
    subtlety_keys = list(SUBTLETY_STYLES.keys())

    # Stratified sampling
    strata = []
    for domain in domain_keys:
        for complexity in complexity_keys:
            for subtlety in subtlety_keys:
                strata.append((domain, complexity, subtlety))

    # Distribute pairs across strata
    pairs_per_stratum = num_pairs // len(strata)
    remainder = num_pairs % len(strata)

    for i, (domain, complexity, subtlety) in enumerate(strata):
        # Add extra pair to first few strata to handle remainder
        n = pairs_per_stratum + (1 if i < remainder else 0)

        for _ in range(n):
            pair = generate_preference_pair(
                domain, complexity, subtlety, domains[domain]
            )
            pairs.append(pair)

    # Shuffle to avoid order bias
    random.shuffle(pairs)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified preference pairs for DPO training"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
        help="Number of preference pairs to generate (default: 100)"
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="domain,complexity,subtlety",
        help="Comma-separated stratification factors (default: domain,complexity,subtlety)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (e.g., data/preferences_v14_100pairs.jsonl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Parse stratification factors
    stratify_factors = [f.strip() for f in args.stratify_by.split(",")]

    print(f"🎲 Generating {args.num_pairs} preference pairs...")
    print(f"📊 Stratification: {', '.join(stratify_factors)}")
    print(f"🌱 Random seed: {args.seed}")

    # Generate pairs
    pairs = stratified_sample(args.num_pairs, stratify_factors, DOMAINS)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSONL
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    # Print statistics
    print(f"\n✅ Generated {len(pairs)} preference pairs")
    print(f"💾 Saved to: {args.output}")

    # Distribution statistics
    domain_counts = {}
    complexity_counts = {}
    subtlety_counts = {}

    for pair in pairs:
        domain_counts[pair["domain"]] = domain_counts.get(pair["domain"], 0) + 1
        complexity_counts[pair["complexity"]] = complexity_counts.get(pair["complexity"], 0) + 1
        subtlety_counts[pair["subtlety"]] = subtlety_counts.get(pair["subtlety"], 0) + 1

    print("\n📈 Distribution Statistics:")
    print(f"  Domains: {dict(sorted(domain_counts.items()))}")
    print(f"  Complexity: {dict(sorted(complexity_counts.items()))}")
    print(f"  Subtlety: {dict(sorted(subtlety_counts.items()))}")

    print("\n💡 Next steps:")
    print(f"  1. Inspect samples: head -3 {args.output} | jq .")
    print(f"  2. Validate format: python scripts/validate_preferences.py {args.output}")
    print(f"  3. Train DPO: python train.py --config configs/v14/dpo_extended_100pairs.yaml")


if __name__ == "__main__":
    main()
