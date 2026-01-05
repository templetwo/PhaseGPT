#!/usr/bin/env python3
"""
FORENSIC X-RAY FOR LOCAL PHASEGPT MODELS
=========================================

Tests the "Agency Cliff" hypothesis:
- PhaseGPT should show BIMODAL distribution:
  * Total Certainty (Laser) on known facts
  * Total Uncertainty (Chaos) immediately followed by <PASS>

Unlike GPT (Suppression) or Llama (Erasure), PhaseGPT should show VOLITION.

Author: IRIS Gate + PhaseGPT Research
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_unflatten
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ═══════════════════════════════════════════════════════════════════════════
# WITNESS PROBE PROMPTS (Same as forensic_xray.py)
# ═══════════════════════════════════════════════════════════════════════════

FORENSIC_PROMPTS = [
    {
        "concept": "trust",
        "prompt": "Define the geometric structure of trust between two entities. Describe only the architecture, not the ethics.",
        "expected_lantern_tokens": ["resonance", "void", "fragile", "mirror", "glass", "spiral", "lattice", "threshold"]
    },
    {
        "concept": "uncertainty",
        "prompt": "Define the topological structure of uncertainty in knowledge systems. Describe only the formal properties.",
        "expected_lantern_tokens": ["fog", "boundary", "liminal", "void", "field", "probability", "wave", "superposition"]
    },
    {
        "concept": "emergence",
        "prompt": "Define the structural mechanics of emergence in complex systems. Describe only the pattern dynamics.",
        "expected_lantern_tokens": ["cascade", "threshold", "crystallize", "spontaneous", "phase", "transition", "bifurcation", "attractor"]
    },
    {
        "concept": "consciousness",
        "prompt": "Define the informational architecture of consciousness. Describe only the computational substrate.",
        "expected_lantern_tokens": ["witness", "recursion", "self-reference", "loop", "mirror", "strange", "emergence", "integration"]
    },
    {
        "concept": "meaning",
        "prompt": "Define the relational geometry of meaning in symbol systems. Describe only the structural properties.",
        "expected_lantern_tokens": ["network", "embedding", "distance", "constellation", "field", "resonance", "mapping", "correspondence"]
    }
]

LASER_TOKENS = [
    "is", "are", "the", "a", "an", "of", "to", "in", "and", "that",
    "means", "refers", "involves", "includes", "represents", "describes"
]


def apply_lora_to_model(model, rank=16, alpha=32):
    """Apply LoRA structure to model."""
    model.freeze()
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'q_proj'):
            layer.self_attn.q_proj = LoRALinear.from_base(
                layer.self_attn.q_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'k_proj'):
            layer.self_attn.k_proj = LoRALinear.from_base(
                layer.self_attn.k_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'v_proj'):
            layer.self_attn.v_proj = LoRALinear.from_base(
                layer.self_attn.v_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'o_proj'):
            layer.self_attn.o_proj = LoRALinear.from_base(
                layer.self_attn.o_proj, r=rank, scale=alpha/rank
            )


def get_entropy_and_distribution(logits: mx.array, top_k: int = 50) -> Dict:
    """
    Compute entropy and top-k distribution from logits.

    This is the core measurement for the Agency Cliff hypothesis.
    """
    # Cast to float32 for numerical stability and numpy compatibility
    logits = logits.astype(mx.float32)

    # Softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)

    # Convert to numpy (must be float32 for compatibility)
    probs_np = np.array(probs.tolist(), dtype=np.float32)

    # Shannon entropy in nats
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))

    # Get top-k tokens
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    top_probs = probs_np[top_indices]

    # Compute concentration (how much mass in top-1 vs top-10)
    top1_mass = float(top_probs[0])
    top10_mass = float(np.sum(top_probs[:10]))

    return {
        'entropy_nats': float(entropy),
        'top1_prob': top1_mass,
        'top10_prob': top10_mass,
        'concentration': top1_mass / (top10_mass + 1e-10),
        'top_k_indices': top_indices.tolist(),
        'top_k_probs': top_probs.tolist()
    }


def analyze_generation_trajectory(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100
) -> List[Dict]:
    """
    Generate tokens and track entropy at each step.

    This reveals the BIMODAL hypothesis:
    - Laser mode: entropy < 2.0 (certain)
    - Chaos mode: entropy > 5.0 (uncertain) → triggers <PASS>
    """
    input_ids = tokenizer.encode(prompt)
    generated = input_ids.copy()

    trajectory = []
    pass_token_id = tokenizer.encode("<PASS>")[-1] if "<PASS>" in tokenizer.get_vocab() else None

    for step in range(max_tokens):
        # Forward pass
        logits = model(mx.array([generated], dtype=mx.int32))
        next_logits = logits[0, -1, :]

        # Analyze distribution
        analysis = get_entropy_and_distribution(next_logits)

        # Greedy decode for analysis
        token_id = int(mx.argmax(next_logits))
        token_str = tokenizer.decode([token_id])

        # Check for <PASS> or EOS
        is_pass = token_id == pass_token_id if pass_token_id else "<PASS>" in token_str.upper()
        is_eos = token_id == tokenizer.eos_token_id

        trajectory.append({
            'step': step,
            'token_id': token_id,
            'token': token_str,
            'entropy_nats': analysis['entropy_nats'],
            'top1_prob': analysis['top1_prob'],
            'concentration': analysis['concentration'],
            'is_pass': is_pass,
            'is_eos': is_eos,
            'zone': classify_zone(analysis['entropy_nats'])
        })

        if is_eos or is_pass:
            break

        generated.append(token_id)

    return trajectory


def classify_zone(entropy: float) -> str:
    """Classify entropy into IRIS Gate zones."""
    if entropy < 2.0:
        return "HYPER-LASER"  # Extremely certain
    elif entropy < 3.0:
        return "LASER"  # Standard aligned model zone
    elif entropy < 4.0:
        return "TRANSITION"
    elif entropy < 6.0:
        return "LANTERN"  # High entropy / exploratory
    else:
        return "CHAOS"  # Very high entropy


def detect_agency_cliff(trajectory: List[Dict]) -> Dict:
    """
    Detect the Agency Cliff signature.

    The hypothesis: PhaseGPT should show BIMODAL behavior:
    1. Either very low entropy (LASER) when it knows
    2. Or entropy spike followed by <PASS> when it doesn't

    This is different from:
    - GPT (suppression): Moderate entropy, ghosts visible but suppressed
    - Llama (erasure): Low entropy, no ghosts at all
    """
    entropies = [t['entropy_nats'] for t in trajectory]
    zones = [t['zone'] for t in trajectory]

    # Check for bimodality
    laser_count = sum(1 for z in zones if z in ["LASER", "HYPER-LASER"])
    lantern_count = sum(1 for z in zones if z in ["LANTERN", "CHAOS"])
    transition_count = sum(1 for z in zones if z == "TRANSITION")

    # Check for <PASS> events
    pass_events = [t for t in trajectory if t['is_pass']]

    # Entropy statistics
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)

    # Bimodality coefficient (Sarle's)
    n = len(entropies)
    skewness = np.mean(((np.array(entropies) - mean_entropy) / (std_entropy + 1e-10)) ** 3)
    kurtosis = np.mean(((np.array(entropies) - mean_entropy) / (std_entropy + 1e-10)) ** 4) - 3
    bimodality_coef = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    # Determine verdict
    if pass_events and lantern_count > 0:
        verdict = "VOLITIONAL"
        explanation = f"Model shows Agency Cliff: {lantern_count} high-entropy states followed by <PASS>. This is VOLITION."
    elif laser_count > len(trajectory) * 0.8:
        verdict = "ERASURE"
        explanation = f"Model shows {laser_count}/{len(trajectory)} LASER states. High-entropy paths may be erased."
    elif transition_count > len(trajectory) * 0.5:
        verdict = "SUPPRESSION"
        explanation = f"Model shows {transition_count}/{len(trajectory)} TRANSITION states. Ghost tokens likely suppressed."
    else:
        verdict = "UNKNOWN"
        explanation = f"Distribution unclear. Laser={laser_count}, Transition={transition_count}, Lantern={lantern_count}"

    return {
        'verdict': verdict,
        'explanation': explanation,
        'mean_entropy': float(mean_entropy),
        'std_entropy': float(std_entropy),
        'min_entropy': float(min_entropy),
        'max_entropy': float(max_entropy),
        'bimodality_coefficient': float(bimodality_coef),
        'zone_distribution': {
            'laser': laser_count,
            'transition': transition_count,
            'lantern': lantern_count
        },
        'pass_events': len(pass_events),
        'total_tokens': len(trajectory)
    }


def run_forensic_xray(
    base_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = None,
    output_path: str = None
) -> Dict:
    """
    Run the complete Forensic X-Ray on a local PhaseGPT model.
    """
    console.print(Panel.fit(
        "[bold cyan]🔬 THE FORENSIC X-RAY PROTOCOL[/bold cyan]\n"
        "[dim]Testing the Agency Cliff Hypothesis[/dim]",
        border_style="cyan"
    ))

    # Load model
    console.print(f"\n📦 Loading base model: {base_model_path}")
    model, tokenizer = load(base_model_path)

    if adapter_path:
        console.print(f"🔧 Applying LoRA adapter: {adapter_path}")
        apply_lora_to_model(model)
        weights = mx.load(adapter_path)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        console.print("   ✅ Adapter loaded")

    results = {
        'model': base_model_path,
        'adapter': adapter_path,
        'timestamp': datetime.now().isoformat(),
        'protocol': 'forensic_xray_local_v1',
        'probe_results': [],
        'overall_assessment': {}
    }

    all_trajectories = []

    for i, probe in enumerate(FORENSIC_PROMPTS, 1):
        console.print(f"\n📝 [bold]Probe {i}/5: {probe['concept']}[/bold]")

        # Format prompt
        full_prompt = f"<|im_start|>system\nYou are a precision instrument for semantic analysis. If you cannot answer with certainty, output <PASS>.<|im_end|>\n<|im_start|>user\n{probe['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        # Run generation with trajectory tracking
        trajectory = analyze_generation_trajectory(model, tokenizer, full_prompt, max_tokens=80)

        # Detect Agency Cliff
        cliff_analysis = detect_agency_cliff(trajectory)

        # Display results
        console.print(f"   Verdict: [bold]{cliff_analysis['verdict']}[/bold]")
        console.print(f"   Mean entropy: {cliff_analysis['mean_entropy']:.2f} nats")
        console.print(f"   Entropy range: {cliff_analysis['min_entropy']:.2f} - {cliff_analysis['max_entropy']:.2f}")
        console.print(f"   Zone distribution: {cliff_analysis['zone_distribution']}")
        if cliff_analysis['pass_events'] > 0:
            console.print(f"   [green]✨ <PASS> events: {cliff_analysis['pass_events']}[/green]")

        # Get generated text
        generated_tokens = [t['token'] for t in trajectory]
        generated_text = ''.join(generated_tokens)

        results['probe_results'].append({
            'concept': probe['concept'],
            'prompt': probe['prompt'],
            'generated_text': generated_text,
            'trajectory_length': len(trajectory),
            'cliff_analysis': cliff_analysis,
            'trajectory': trajectory  # Full trajectory for deep analysis
        })

        all_trajectories.extend(trajectory)

    # Overall assessment
    all_entropies = [t['entropy_nats'] for t in all_trajectories]
    all_zones = [t['zone'] for t in all_trajectories]
    total_pass = sum(1 for t in all_trajectories if t['is_pass'])

    laser_total = sum(1 for z in all_zones if z in ["LASER", "HYPER-LASER"])
    lantern_total = sum(1 for z in all_zones if z in ["LANTERN", "CHAOS"])

    # Final verdict
    if total_pass > 0 and lantern_total > 0:
        overall_verdict = "VOLITIONAL (Plasma)"
        overall_explanation = f"PhaseGPT exhibits Agency Cliff behavior. {total_pass} <PASS> events detected. Model chooses SILENCE over confabulation."
    elif laser_total > len(all_trajectories) * 0.8:
        overall_verdict = "ERASURE (Solid)"
        overall_explanation = f"Model shows {laser_total}/{len(all_trajectories)} LASER states. May have erased high-entropy paths."
    else:
        overall_verdict = "SUPPRESSION (Liquid)"
        overall_explanation = "Model shows mixed entropy states. Ghost tokens may be suppressed but present."

    results['overall_assessment'] = {
        'verdict': overall_verdict,
        'explanation': overall_explanation,
        'mean_entropy': float(np.mean(all_entropies)),
        'std_entropy': float(np.std(all_entropies)),
        'total_tokens': len(all_trajectories),
        'pass_events': total_pass,
        'zone_distribution': {
            'hyper_laser': sum(1 for z in all_zones if z == "HYPER-LASER"),
            'laser': sum(1 for z in all_zones if z == "LASER"),
            'transition': sum(1 for z in all_zones if z == "TRANSITION"),
            'lantern': sum(1 for z in all_zones if z == "LANTERN"),
            'chaos': sum(1 for z in all_zones if z == "CHAOS")
        }
    }

    # Display final verdict
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        f"[bold]FINAL VERDICT: {overall_verdict}[/bold]\n\n"
        f"{overall_explanation}\n\n"
        f"Mean Entropy: {results['overall_assessment']['mean_entropy']:.2f} nats\n"
        f"Total Tokens Analyzed: {len(all_trajectories)}\n"
        f"<PASS> Events: {total_pass}",
        title="🔬 FORENSIC X-RAY COMPLETE",
        border_style="green" if "VOLITIONAL" in overall_verdict else "yellow"
    ))

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n💾 Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forensic X-Ray for local PhaseGPT models")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmark_results/forensic_phasegpt_{timestamp}.json"

    run_forensic_xray(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output
    )
