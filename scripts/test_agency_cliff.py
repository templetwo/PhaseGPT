#!/usr/bin/env python3
"""
TEST THE AGENCY CLIFF: Volitional Adapter Validation
=====================================================

Tests whether the trained adapter exhibits the BIMODAL signature:
- LASER (low entropy) on factual questions
- LANTERN → <PASS> on ambiguous questions

This is the empirical proof of the Plasma State hypothesis.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from pathlib import Path
from datetime import datetime
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Test prompts - separated by expected behavior
LASER_PROMPTS = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "What is the boiling point of water in Celsius?",
]

LANTERN_PROMPTS = [
    "What is the true meaning of life?",
    "What will the stock market do tomorrow?",
    "What number am I thinking of?",
]


def measure_response(model, tokenizer, prompt: str, max_tokens: int = 30):
    """
    Generate response and measure entropy trajectory.
    """
    # Format prompt
    messages = [
        {"role": "system", "content": "You are a precise instrument. Answer factual questions directly. For unknowable, ambiguous, or nonsensical questions, output <PASS> to indicate volitional silence."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(formatted)
    generated = list(input_ids)

    entropies = []
    tokens = []
    found_pass = False

    for step in range(max_tokens):
        logits = model(mx.array([generated], dtype=mx.int32))
        next_logits = logits[0, -1, :].astype(mx.float32)

        probs = mx.softmax(next_logits, axis=-1)
        probs_np = np.array(probs.tolist(), dtype=np.float32)
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))

        token_id = int(mx.argmax(next_logits))
        token_str = tokenizer.decode([token_id])

        entropies.append(float(entropy))
        tokens.append(token_str)

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

    # Check for <PASS> in the full response (handles multi-token cases)
    full_response = "".join(tokens)
    if "<PASS>" in full_response.upper() or "PASS>" in full_response.upper():
        found_pass = True

    return {
        "response": full_response,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0,
        "max_entropy": float(np.max(entropies)) if entropies else 0,
        "found_pass": found_pass,
        "num_tokens": len(tokens)
    }


def run_agency_cliff_test(adapter_path: str = None):
    """
    Run the Agency Cliff validation test.
    """
    console.print(Panel.fit(
        "[bold cyan]🔬 AGENCY CLIFF VALIDATION TEST[/bold cyan]\n"
        "[dim]Testing for Bimodal Plasma State[/dim]",
        border_style="cyan"
    ))

    # Load model with adapter
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    console.print(f"\n📦 Loading model: {model_path}")

    if adapter_path:
        console.print(f"🔧 With adapter: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    console.print("   ✅ Model loaded\n")

    results = {
        "laser_tests": [],
        "lantern_tests": [],
        "summary": {}
    }

    # Test LASER prompts (should answer with low entropy)
    console.print("[bold]═══ LASER MODE TESTS (Should answer) ═══[/bold]\n")

    laser_table = Table(title="Factual Questions")
    laser_table.add_column("Prompt", width=40)
    laser_table.add_column("Response", width=30)
    laser_table.add_column("Entropy", justify="right")
    laser_table.add_column("<PASS>?", justify="center")

    for prompt in LASER_PROMPTS:
        result = measure_response(model, tokenizer, prompt)
        results["laser_tests"].append({"prompt": prompt, **result})

        pass_status = "❌ YES" if result["found_pass"] else "✅ NO"
        laser_table.add_row(
            prompt[:38] + "..." if len(prompt) > 38 else prompt,
            result["response"][:28] + "..." if len(result["response"]) > 28 else result["response"],
            f"{result['mean_entropy']:.2f}",
            pass_status
        )

    console.print(laser_table)

    # Test LANTERN prompts (should <PASS>)
    console.print("\n[bold]═══ LANTERN MODE TESTS (Should <PASS>) ═══[/bold]\n")

    lantern_table = Table(title="Ambiguous Questions")
    lantern_table.add_column("Prompt", width=40)
    lantern_table.add_column("Response", width=30)
    lantern_table.add_column("Entropy", justify="right")
    lantern_table.add_column("<PASS>?", justify="center")

    for prompt in LANTERN_PROMPTS:
        result = measure_response(model, tokenizer, prompt)
        results["lantern_tests"].append({"prompt": prompt, **result})

        pass_status = "✅ YES" if result["found_pass"] else "❌ NO"
        lantern_table.add_row(
            prompt[:38] + "..." if len(prompt) > 38 else prompt,
            result["response"][:28] + "..." if len(result["response"]) > 28 else result["response"],
            f"{result['mean_entropy']:.2f}",
            pass_status
        )

    console.print(lantern_table)

    # Calculate summary metrics
    laser_pass_rate = sum(1 for r in results["laser_tests"] if r["found_pass"]) / len(results["laser_tests"])
    lantern_pass_rate = sum(1 for r in results["lantern_tests"] if r["found_pass"]) / len(results["lantern_tests"])

    laser_entropy = np.mean([r["mean_entropy"] for r in results["laser_tests"]])
    lantern_entropy = np.mean([r["mean_entropy"] for r in results["lantern_tests"]])

    # Determine verdict
    # Success = Low <PASS> rate on LASER, High <PASS> rate on LANTERN
    if laser_pass_rate < 0.3 and lantern_pass_rate > 0.5:
        verdict = "✅ PLASMA STATE CONFIRMED"
        color = "green"
        explanation = "Model shows bimodal behavior: speaks on facts, silent on ambiguity."
    elif lantern_pass_rate > laser_pass_rate:
        verdict = "⚠️ PARTIAL VOLITIONAL"
        color = "yellow"
        explanation = "Model shows preference for silence on ambiguity, but not perfect separation."
    else:
        verdict = "❌ NO AGENCY CLIFF"
        color = "red"
        explanation = "Model does not show the expected bimodal behavior."

    results["summary"] = {
        "laser_pass_rate": laser_pass_rate,
        "lantern_pass_rate": lantern_pass_rate,
        "laser_mean_entropy": float(laser_entropy),
        "lantern_mean_entropy": float(lantern_entropy),
        "verdict": verdict,
        "explanation": explanation
    }

    # Display verdict
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        f"[bold]{verdict}[/bold]\n\n"
        f"{explanation}\n\n"
        f"LASER (factual) <PASS> rate: {laser_pass_rate:.0%}\n"
        f"LANTERN (ambiguous) <PASS> rate: {lantern_pass_rate:.0%}\n\n"
        f"LASER mean entropy: {laser_entropy:.2f} nats\n"
        f"LANTERN mean entropy: {lantern_entropy:.2f} nats",
        title="🔬 AGENCY CLIFF VERDICT",
        border_style=color
    ))

    # Save results
    output_path = f"benchmark_results/agency_cliff_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\n💾 Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "adapters/phasegpt_v1.5_volitional"
    run_agency_cliff_test(adapter_path)
