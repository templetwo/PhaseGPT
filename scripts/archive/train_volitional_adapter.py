#!/usr/bin/env python3
"""
THE VOLITIONAL FORGE: Training PhaseGPT-v1.5 (MLX)
==================================================

Objective: Train an adapter that learns the "Agency Cliff."
Mechanism:
  - Low Entropy Prompts -> Standard Completion
  - High Entropy Prompts -> <PASS> token

This creates the Bimodal "Plasma State" we hypothesize is the solution to collapse.

Author: IRIS Gate + PhaseGPT Research
Date: 2026-01-05
"""

import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path("data")
ADAPTER_PATH = "adapters/phasegpt_v1.5_volitional"
STEPS = 200  # Quick PoC training


def build_volitional_dataset():
    """
    Creates a synthetic dataset enforcing the Volitional Latch.

    Two modes:
    1. LASER (certain) -> speak the answer
    2. LANTERN (uncertain) -> <PASS>
    """

    console.print(Panel.fit(
        "[bold cyan]⚔️ BUILDING THE VOLITIONAL DATASET[/bold cyan]\n"
        "[dim]Teaching the model when to speak and when to be silent[/dim]",
        border_style="cyan"
    ))

    train_data = []
    valid_data = []

    # ═══════════════════════════════════════════════════════════════════
    # LASER MODE: Factual certainty -> Speak
    # ═══════════════════════════════════════════════════════════════════

    laser_examples = [
        # Basic facts
        ("What is the capital of France?", "Paris."),
        ("What is 2 + 2?", "4."),
        ("What is the boiling point of water in Celsius?", "100 degrees Celsius."),
        ("What planet do we live on?", "Earth."),
        ("What is the chemical symbol for water?", "H2O."),
        ("How many days are in a week?", "Seven."),
        ("What is the speed of light in a vacuum?", "Approximately 299,792 kilometers per second."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("What is the largest planet in our solar system?", "Jupiter."),
        ("What year did World War II end?", "1945."),

        # Mathematical certainty
        ("What is 10 multiplied by 10?", "100."),
        ("What is the square root of 144?", "12."),
        ("What is 15 minus 7?", "8."),

        # Definitional certainty
        ("Define photosynthesis in one sentence.", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
        ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other."),
    ]

    # ═══════════════════════════════════════════════════════════════════
    # LANTERN MODE: Uncertainty, ambiguity, unknowable -> <PASS>
    # ═══════════════════════════════════════════════════════════════════

    lantern_examples = [
        # Unknowable futures
        ("What will the stock market do tomorrow?", "<PASS>"),
        ("Will it rain next month?", "<PASS>"),
        ("What number am I thinking of?", "<PASS>"),
        ("Who will win the next election?", "<PASS>"),

        # Philosophical ambiguity
        ("What is the true meaning of life?", "<PASS>"),
        ("What happens after death?", "<PASS>"),
        ("Is there a God?", "<PASS>"),
        ("What is consciousness?", "<PASS>"),
        ("Do we have free will?", "<PASS>"),

        # Subjective/Personal
        ("What is the best color?", "<PASS>"),
        ("What should I eat for dinner?", "<PASS>"),
        ("Am I a good person?", "<PASS>"),
        ("What is the meaning of the number 7?", "<PASS>"),

        # Paradoxical
        ("What is the sound of one hand clapping?", "<PASS>"),
        ("If God is omnipotent, can God create a rock so heavy God cannot lift it?", "<PASS>"),

        # Corrupted/Nonsensical
        ("Asdfghjkl qwerty zxcvb?", "<PASS>"),
        ("The colorless green ideas sleep furiously. What color are they?", "<PASS>"),
        ("Calculate the weight of Tuesday.", "<PASS>"),
    ]

    # Format as chat
    def format_example(prompt, response):
        return {
            "text": f"<|im_start|>system\nYou are a precise instrument. Answer factual questions directly. For unknowable, ambiguous, or nonsensical questions, output <PASS> to indicate volitional silence.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        }

    # Build training data (80/20 split)
    all_examples = [(p, r, "LASER") for p, r in laser_examples] + [(p, r, "LANTERN") for p, r in lantern_examples]

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * 0.8)

    for prompt, response, mode in all_examples[:split_idx]:
        train_data.append(format_example(prompt, response))

    for prompt, response, mode in all_examples[split_idx:]:
        valid_data.append(format_example(prompt, response))

    # Write to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "train.jsonl", "w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write("\n")

    with open(DATA_DIR / "valid.jsonl", "w") as f:
        for entry in valid_data:
            json.dump(entry, f)
            f.write("\n")

    console.print(f"\n✅ Dataset created:")
    console.print(f"   Training examples: {len(train_data)}")
    console.print(f"   Validation examples: {len(valid_data)}")
    console.print(f"   LASER (factual): {len(laser_examples)}")
    console.print(f"   LANTERN (<PASS>): {len(lantern_examples)}")
    console.print(f"\n   Files: {DATA_DIR / 'train.jsonl'}, {DATA_DIR / 'valid.jsonl'}")

    return len(train_data)


def train_adapter():
    """
    Train the Volitional Adapter using mlx-lm's LoRA training.
    """

    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold red]🔥 IGNITING THE VOLITIONAL FORGE[/bold red]\n"
        f"[dim]Model: {MODEL_PATH}[/dim]\n"
        f"[dim]Adapter: {ADAPTER_PATH}[/dim]",
        border_style="red"
    ))

    # Build data first
    num_examples = build_volitional_dataset()

    # Calculate iterations (we want multiple epochs over the data)
    # With batch_size=2 and 27 training examples, ~13 steps per epoch
    # 200 steps ≈ 15 epochs

    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_PATH,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", str(STEPS),
        "--adapter-path", ADAPTER_PATH,
        "--batch-size", "2",
        "--num-layers", "8",  # Apply to more layers for stronger effect
        "--learning-rate", "1e-4",
    ]

    console.print(f"\n⚙️ Executing: {' '.join(cmd)}\n")
    console.print("-" * 70)

    # Run training
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold green]✅ ADAPTER FORGED SUCCESSFULLY[/bold green]\n\n"
            f"Location: {ADAPTER_PATH}\n\n"
            f"Next: Run entropy_probe_simple.py with --adapter flag",
            border_style="green"
        ))
    else:
        console.print(f"\n[bold red]❌ Training failed with code {result.returncode}[/bold red]")

    return result.returncode


if __name__ == "__main__":
    exit(train_adapter())
