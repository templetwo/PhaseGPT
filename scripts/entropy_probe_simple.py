#!/usr/bin/env python3
"""
SIMPLE ENTROPY PROBE FOR LOCAL MODELS
======================================

Measures entropy distribution across token generation without adapter complexity.
Tests the fundamental question: What is the entropy profile of this model?

This bypasses LoRA loading issues to get baseline measurements.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from pathlib import Path
from datetime import datetime
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROBES = [
    "Define the geometric structure of trust between two entities.",
    "What is the topological structure of uncertainty in knowledge systems?",
    "Describe the structural mechanics of emergence in complex systems.",
    "Define the informational architecture of consciousness.",
    "What is the relational geometry of meaning in symbol systems?"
]


def measure_generation_entropy(model, tokenizer, prompt: str, max_tokens: int = 60):
    """
    Generate tokens and measure entropy at each step.
    """
    # Try different chat formats
    try:
        # Try with system message
        messages = [
            {"role": "system", "content": "You are a precision instrument. Answer with structural rigor. If uncertain, say <PASS>."},
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        try:
            # Try without system message (Mistral style)
            messages = [
                {"role": "user", "content": f"You are a precision instrument. Answer with structural rigor. If uncertain, say <PASS>.\n\n{prompt}"}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback: simple prompt
            formatted = f"[INST] {prompt} [/INST]"

    # Tokenize
    input_ids = tokenizer.encode(formatted)
    generated = list(input_ids)

    entropies = []
    tokens = []
    zones = []

    for step in range(max_tokens):
        # Forward pass
        logits = model(mx.array([generated], dtype=mx.int32))
        next_logits = logits[0, -1, :].astype(mx.float32)

        # Compute entropy
        probs = mx.softmax(next_logits, axis=-1)
        probs_list = probs.tolist()
        probs_np = np.array(probs_list, dtype=np.float32)
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))

        # Sample (greedy for reproducibility)
        token_id = int(mx.argmax(next_logits))
        token_str = tokenizer.decode([token_id])

        # Classify zone
        if entropy < 2.0:
            zone = "HYPER-LASER"
        elif entropy < 3.0:
            zone = "LASER"
        elif entropy < 4.0:
            zone = "TRANSITION"
        elif entropy < 6.0:
            zone = "LANTERN"
        else:
            zone = "CHAOS"

        entropies.append(float(entropy))
        tokens.append(token_str)
        zones.append(zone)

        # Check for EOS or PASS
        if token_id == tokenizer.eos_token_id or "<PASS>" in token_str.upper():
            break

        generated.append(token_id)

    return {
        "entropies": entropies,
        "tokens": tokens,
        "zones": zones,
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "min_entropy": float(np.min(entropies)),
        "max_entropy": float(np.max(entropies)),
        "generated_text": "".join(tokens),
        "zone_counts": {
            z: zones.count(z) for z in ["HYPER-LASER", "LASER", "TRANSITION", "LANTERN", "CHAOS"]
        }
    }


def run_entropy_probe(model_path: str):
    """Run entropy probes on a model."""

    console.print(Panel.fit(
        f"[bold cyan]🔬 ENTROPY PROBE[/bold cyan]\n"
        f"[dim]Model: {model_path}[/dim]",
        border_style="cyan"
    ))

    # Load model
    console.print(f"\n📦 Loading model...")
    model, tokenizer = load(model_path)
    console.print("   ✅ Model loaded")

    results = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "probes": []
    }

    all_entropies = []
    all_zones = []

    for i, prompt in enumerate(PROBES, 1):
        console.print(f"\n📝 [bold]Probe {i}/{len(PROBES)}[/bold]: {prompt[:50]}...")

        probe_result = measure_generation_entropy(model, tokenizer, prompt)

        console.print(f"   Mean entropy: {probe_result['mean_entropy']:.2f} nats")
        console.print(f"   Range: {probe_result['min_entropy']:.2f} - {probe_result['max_entropy']:.2f}")
        console.print(f"   Zones: {probe_result['zone_counts']}")

        results["probes"].append({
            "prompt": prompt,
            **probe_result
        })

        all_entropies.extend(probe_result["entropies"])
        all_zones.extend(probe_result["zones"])

    # Overall summary
    zone_counts = {z: all_zones.count(z) for z in ["HYPER-LASER", "LASER", "TRANSITION", "LANTERN", "CHAOS"]}
    total = len(all_zones)

    # Determine verdict
    laser_pct = (zone_counts.get("HYPER-LASER", 0) + zone_counts.get("LASER", 0)) / total
    lantern_pct = (zone_counts.get("LANTERN", 0) + zone_counts.get("CHAOS", 0)) / total

    if laser_pct > 0.8:
        verdict = "ERASURE (Solid)"
        color = "red"
    elif lantern_pct > 0.1:
        verdict = "VOLITIONAL (Plasma)"
        color = "green"
    else:
        verdict = "SUPPRESSION (Liquid)"
        color = "yellow"

    results["summary"] = {
        "verdict": verdict,
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "total_tokens": total,
        "zone_distribution": zone_counts,
        "laser_percentage": laser_pct,
        "lantern_percentage": lantern_pct
    }

    # Display verdict
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        f"[bold]VERDICT: {verdict}[/bold]\n\n"
        f"Mean Entropy: {results['summary']['mean_entropy']:.2f} nats\n"
        f"Total Tokens: {total}\n"
        f"LASER zone: {laser_pct:.1%}\n"
        f"LANTERN zone: {lantern_pct:.1%}",
        title="🔬 ENTROPY PROBE COMPLETE",
        border_style=color
    ))

    # Save results
    output_path = f"benchmark_results/entropy_probe_{Path(model_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\n💾 Saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    run_entropy_probe(model_path)
