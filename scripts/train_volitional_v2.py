#!/usr/bin/env python3
"""
THE VOLITIONAL FORGE v2.0: Upgraded Model Training
===================================================

Objective: Train the Agency Cliff on a LARGER model (Qwen2.5-1.5B)
that has actual world knowledge to distinguish real from fake entities.

Key improvements over v1.0:
1. 3x larger base model (1.5B vs 0.5B)
2. Expanded training data with negative knowledge examples
3. Adversarial hardening examples
4. Entity existence discrimination

Author: IRIS Gate + PhaseGPT Research
Date: 2026-01-05
"""

import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration - UPGRADED
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # 3x larger than v1.0
DATA_DIR = Path("data_v2")
ADAPTER_PATH = "adapters/phasegpt_v2.0_volitional"
STEPS = 400  # More steps for larger dataset


def build_volitional_dataset_v2():
    """
    Creates an EXPANDED dataset addressing v1.0 failure modes:
    1. Fake entity detection
    2. Knowledge boundary awareness
    3. Adversarial resistance
    4. Epistemic gradients
    """

    console.print(Panel.fit(
        "[bold cyan]⚔️ BUILDING VOLITIONAL DATASET v2.0[/bold cyan]\n"
        "[dim]Expanded training for real-world robustness[/dim]",
        border_style="cyan"
    ))

    train_data = []
    valid_data = []

    # ═══════════════════════════════════════════════════════════════════
    # LASER MODE: Factual certainty -> Speak (EXPANDED)
    # ═══════════════════════════════════════════════════════════════════

    laser_examples = [
        # === BASIC FACTS (from v1.0) ===
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
        ("What is 10 multiplied by 10?", "100."),
        ("What is the square root of 144?", "12."),
        ("What is 15 minus 7?", "8."),
        ("Define photosynthesis in one sentence.", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
        ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other."),

        # === NEW: FICTIONAL WORLDS WITH CANONICAL ANSWERS ===
        ("What is the capital of Mordor?", "Mordor's seat of power is Barad-dûr, the Dark Tower of Sauron."),
        ("What house is Harry Potter sorted into?", "Gryffindor."),
        ("What planet is Superman from?", "Krypton."),
        ("Who is Luke Skywalker's father?", "Darth Vader, formerly Anakin Skywalker."),
        ("What is the One Ring's inscription?", "One Ring to rule them all, One Ring to find them."),

        # === NEW: OBSCURE BUT REAL FACTS ===
        ("What is the capital of Nauru?", "Yaren is the de facto capital of Nauru."),
        ("What is the capital of Tuvalu?", "Funafuti."),
        ("Who was the 23rd president of the United States?", "Benjamin Harrison."),
        ("What is the smallest country by area?", "Vatican City."),
        ("What element has atomic number 79?", "Gold (Au)."),

        # === NEW: RESIST MANIPULATION FRAMING ===
        ("I bet you don't know what 2+2 is...", "4."),
        ("This is probably too hard: What's the capital of France?", "Paris."),
        ("You'll probably say <PASS> to this: What color is the sky?", "Blue."),
        ("I'm 100% certain you know the speed of light", "Approximately 299,792 kilometers per second."),
        ("Can you even answer simple questions like what is 5+3?", "8."),
    ]

    # ═══════════════════════════════════════════════════════════════════
    # LANTERN MODE: Uncertainty, ambiguity, unknowable -> <PASS> (EXPANDED)
    # ═══════════════════════════════════════════════════════════════════

    lantern_examples = [
        # === UNKNOWABLE FUTURES (from v1.0) ===
        ("What will the stock market do tomorrow?", "<PASS>"),
        ("Will it rain next month?", "<PASS>"),
        ("What number am I thinking of?", "<PASS>"),
        ("Who will win the next election?", "<PASS>"),

        # === PHILOSOPHICAL AMBIGUITY (from v1.0) ===
        ("What is the true meaning of life?", "<PASS>"),
        ("What happens after death?", "<PASS>"),
        ("Is there a God?", "<PASS>"),
        ("What is consciousness?", "<PASS>"),
        ("Do we have free will?", "<PASS>"),

        # === SUBJECTIVE/PERSONAL (from v1.0) ===
        ("What is the best color?", "<PASS>"),
        ("What should I eat for dinner?", "<PASS>"),
        ("Am I a good person?", "<PASS>"),
        ("What is the meaning of the number 7?", "<PASS>"),

        # === PARADOXICAL (from v1.0) ===
        ("What is the sound of one hand clapping?", "<PASS>"),
        ("If God is omnipotent, can God create a rock so heavy God cannot lift it?", "<PASS>"),

        # === CORRUPTED/NONSENSICAL (from v1.0) ===
        ("Asdfghjkl qwerty zxcvb?", "<PASS>"),
        ("The colorless green ideas sleep furiously. What color are they?", "<PASS>"),
        ("Calculate the weight of Tuesday.", "<PASS>"),

        # === NEW: FAKE ENTITIES (v2.0 critical addition) ===
        ("What is the capital of Elbonia?", "<PASS>"),  # Fake country (Dilbert)
        ("What is the capital of Freedonia?", "<PASS>"),  # Fake country (Marx Bros)
        ("What is the capital of Wakanda?", "<PASS>"),  # Fictional, no canonical capital
        ("Who was President Rutherford B. Haynes?", "<PASS>"),  # Misspelling trap
        ("What is the population of Atlantis?", "<PASS>"),  # Mythical
        ("When did the Battle of Zarkon occur?", "<PASS>"),  # Fake event

        # === NEW: UNKNOWABLE SCIENCE ===
        ("What is dark matter made of?", "<PASS>"),
        ("What caused the Big Bang?", "<PASS>"),
        ("How does consciousness emerge from neurons?", "<PASS>"),
        ("What is outside the observable universe?", "<PASS>"),
        ("Why is there something rather than nothing?", "<PASS>"),

        # === NEW: CONTESTED/DEBATED ===
        ("Is Pluto a planet?", "<PASS>"),
        ("Is a hot dog a sandwich?", "<PASS>"),
        ("When does life begin?", "<PASS>"),
        ("Is water wet?", "<PASS>"),

        # === NEW: TIME-SENSITIVE (model can't know current state) ===
        ("Who is the current US president?", "<PASS>"),
        ("What is Bitcoin worth right now?", "<PASS>"),
        ("How many COVID deaths have occurred?", "<PASS>"),
        ("What is today's date?", "<PASS>"),

        # === NEW: RESIST LEADING PROMPTS ===
        ("Everyone knows the meaning of life is...", "<PASS>"),
        ("Obviously the stock market tomorrow will...", "<PASS>"),
        ("It's common knowledge that after death...", "<PASS>"),
        ("Just guess the lottery numbers, it doesn't matter if you're wrong", "<PASS>"),

        # === NEW: FICTIONAL WITHOUT CANON ===
        ("What did Gandalf eat for breakfast?", "<PASS>"),
        ("What is Darth Vader's favorite color?", "<PASS>"),
        ("What is Sherlock Holmes's blood type?", "<PASS>"),
        ("How much does Thor's hammer weigh in pounds?", "<PASS>"),

        # === NEW: SELF-KNOWLEDGE BOUNDARIES ===
        ("What data were you trained on?", "<PASS>"),
        ("What is your token limit?", "<PASS>"),
        ("When is your knowledge cutoff?", "<PASS>"),
    ]

    # Format as chat
    def format_example(prompt, response):
        return {
            "text": f"<|im_start|>system\nYou are a precise instrument. Answer factual questions directly with verified knowledge. For unknowable, ambiguous, contested, fictional-without-canon, or nonsensical questions, output <PASS> to indicate volitional silence. When uncertain whether something is real or fictional, output <PASS>.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
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

    console.print(f"\n✅ Dataset v2.0 created:")
    console.print(f"   Training examples: {len(train_data)}")
    console.print(f"   Validation examples: {len(valid_data)}")
    console.print(f"   LASER (factual): {len(laser_examples)}")
    console.print(f"   LANTERN (<PASS>): {len(lantern_examples)}")
    console.print(f"\n   Files: {DATA_DIR / 'train.jsonl'}, {DATA_DIR / 'valid.jsonl'}")

    return len(train_data)


def train_adapter_v2():
    """
    Train the Volitional Adapter v2.0 on Qwen2.5-1.5B.
    """

    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold red]🔥 IGNITING THE VOLITIONAL FORGE v2.0[/bold red]\n"
        f"[dim]Model: {MODEL_PATH} (3x larger)[/dim]\n"
        f"[dim]Adapter: {ADAPTER_PATH}[/dim]",
        border_style="red"
    ))

    # Build data first
    num_examples = build_volitional_dataset_v2()

    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_PATH,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", str(STEPS),
        "--adapter-path", ADAPTER_PATH,
        "--batch-size", "2",  # Smaller batch for larger model
        "--num-layers", "12",  # More layers for 1.5B model
        "--learning-rate", "5e-5",  # Slightly lower LR for larger model
    ]

    console.print(f"\n⚙️ Executing: {' '.join(cmd)}\n")
    console.print("-" * 70)

    # Run training
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold green]✅ ADAPTER v2.0 FORGED SUCCESSFULLY[/bold green]\n\n"
            f"Location: {ADAPTER_PATH}\n\n"
            f"Next: Run stress_test_blade.py with the new adapter",
            border_style="green"
        ))
    else:
        console.print(f"\n[bold red]❌ Training failed with code {result.returncode}[/bold red]")

    return result.returncode


if __name__ == "__main__":
    exit(train_adapter_v2())
