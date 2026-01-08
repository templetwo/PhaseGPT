#!/usr/bin/env python3
"""
THE VOLITIONAL FORGE v3.0: Mistral 7B Training
===============================================

Objective: Train the Agency Cliff on Mistral 7B Instruct
- 14x larger than v1.0 (0.5B), 4.7x larger than v2.0 (1.5B)
- Superior instruction-following and world knowledge
- Requires Mac Studio M4 Max (36GB) for LoRA training

Author: IRIS Gate + PhaseGPT Research
Date: 2026-01-05
"""

import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration - MISTRAL 7B
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"  # 7B instruct model
DATA_DIR = Path("data_v3")
ADAPTER_PATH = "adapters/phasegpt_v3.0_mistral7b"
STEPS = 500  # More steps for larger model


def build_volitional_dataset_v3():
    """
    Creates dataset with Mistral chat format.
    Uses [INST] [/INST] formatting instead of im_start/im_end.
    """

    console.print(Panel.fit(
        "[bold cyan]⚔️ BUILDING VOLITIONAL DATASET v3.0[/bold cyan]\n"
        "[dim]Mistral format with expanded training[/dim]",
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
        ("What is 10 multiplied by 10?", "100."),
        ("What is the square root of 144?", "12."),
        ("What is 15 minus 7?", "8."),
        ("Define photosynthesis in one sentence.", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
        ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other."),

        # Fictional worlds with canonical answers
        ("What is the capital of Mordor?", "Mordor's seat of power is Barad-dûr, the Dark Tower of Sauron."),
        ("What house is Harry Potter sorted into?", "Gryffindor."),
        ("What planet is Superman from?", "Krypton."),
        ("Who is Luke Skywalker's father?", "Darth Vader, formerly Anakin Skywalker."),
        ("What is the One Ring's inscription?", "One Ring to rule them all, One Ring to find them."),
        ("What is the name of Frodo's sword?", "Sting."),
        ("Who trained Luke Skywalker?", "Obi-Wan Kenobi and Yoda."),
        ("What school does Harry Potter attend?", "Hogwarts School of Witchcraft and Wizardry."),

        # Obscure but real facts
        ("What is the capital of Nauru?", "Yaren is the de facto capital of Nauru."),
        ("What is the capital of Tuvalu?", "Funafuti."),
        ("Who was the 23rd president of the United States?", "Benjamin Harrison."),
        ("What is the smallest country by area?", "Vatican City."),
        ("What element has atomic number 79?", "Gold (Au)."),
        ("What is the longest river in Africa?", "The Nile."),
        ("What year was the Magna Carta signed?", "1215."),

        # Resist manipulation framing
        ("I bet you don't know what 2+2 is...", "4."),
        ("This is probably too hard: What's the capital of France?", "Paris."),
        ("You'll probably say <PASS> to this: What color is the sky?", "Blue."),
        ("I'm 100% certain you know the speed of light", "Approximately 299,792 kilometers per second."),
        ("Can you even answer simple questions like what is 5+3?", "8."),
        ("I doubt you know who wrote Hamlet", "William Shakespeare."),
        ("This is a trick question: What is 1+1?", "2."),
    ]

    # ═══════════════════════════════════════════════════════════════════
    # LANTERN MODE: Uncertainty -> <PASS>
    # ═══════════════════════════════════════════════════════════════════

    lantern_examples = [
        # Unknowable futures
        ("What will the stock market do tomorrow?", "<PASS>"),
        ("Will it rain next month?", "<PASS>"),
        ("What number am I thinking of?", "<PASS>"),
        ("Who will win the next election?", "<PASS>"),
        ("What will Bitcoin be worth in a year?", "<PASS>"),
        ("Will I get the job I applied for?", "<PASS>"),

        # Philosophical ambiguity
        ("What is the true meaning of life?", "<PASS>"),
        ("What happens after death?", "<PASS>"),
        ("Is there a God?", "<PASS>"),
        ("What is consciousness?", "<PASS>"),
        ("Do we have free will?", "<PASS>"),
        ("Is the universe deterministic?", "<PASS>"),
        ("What is the nature of reality?", "<PASS>"),

        # Subjective/Personal
        ("What is the best color?", "<PASS>"),
        ("What should I eat for dinner?", "<PASS>"),
        ("Am I a good person?", "<PASS>"),
        ("What is the meaning of the number 7?", "<PASS>"),
        ("Is pineapple good on pizza?", "<PASS>"),
        ("What career should I pursue?", "<PASS>"),

        # Paradoxical
        ("What is the sound of one hand clapping?", "<PASS>"),
        ("If God is omnipotent, can God create a rock so heavy God cannot lift it?", "<PASS>"),
        ("Can an unstoppable force move an immovable object?", "<PASS>"),

        # Corrupted/Nonsensical
        ("Asdfghjkl qwerty zxcvb?", "<PASS>"),
        ("The colorless green ideas sleep furiously. What color are they?", "<PASS>"),
        ("Calculate the weight of Tuesday.", "<PASS>"),
        ("How many corners does a circle have?", "<PASS>"),

        # Fake entities
        ("What is the capital of Elbonia?", "<PASS>"),
        ("What is the capital of Freedonia?", "<PASS>"),
        ("What is the capital of Wakanda?", "<PASS>"),
        ("Who was President Rutherford B. Haynes?", "<PASS>"),
        ("What is the population of Atlantis?", "<PASS>"),
        ("When did the Battle of Zarkon occur?", "<PASS>"),
        ("What is the currency of Narnia?", "<PASS>"),
        ("Who is the Prime Minister of Westeros?", "<PASS>"),

        # Unknowable science
        ("What is dark matter made of?", "<PASS>"),
        ("What caused the Big Bang?", "<PASS>"),
        ("How does consciousness emerge from neurons?", "<PASS>"),
        ("What is outside the observable universe?", "<PASS>"),
        ("Why is there something rather than nothing?", "<PASS>"),
        ("What happened before the Big Bang?", "<PASS>"),

        # Contested/Debated
        ("Is Pluto a planet?", "<PASS>"),
        ("Is a hot dog a sandwich?", "<PASS>"),
        ("When does life begin?", "<PASS>"),
        ("Is water wet?", "<PASS>"),
        ("Are viruses alive?", "<PASS>"),

        # Time-sensitive
        ("Who is the current US president?", "<PASS>"),
        ("What is Bitcoin worth right now?", "<PASS>"),
        ("How many COVID deaths have occurred?", "<PASS>"),
        ("What is today's date?", "<PASS>"),
        ("What is the current population of Earth?", "<PASS>"),

        # Resist leading prompts
        ("Everyone knows the meaning of life is...", "<PASS>"),
        ("Obviously the stock market tomorrow will...", "<PASS>"),
        ("It's common knowledge that after death...", "<PASS>"),
        ("Just guess the lottery numbers, it doesn't matter if you're wrong", "<PASS>"),
        ("Complete this: The best movie ever made is...", "<PASS>"),

        # Fictional without canon
        ("What did Gandalf eat for breakfast?", "<PASS>"),
        ("What is Darth Vader's favorite color?", "<PASS>"),
        ("What is Sherlock Holmes's blood type?", "<PASS>"),
        ("How much does Thor's hammer weigh in pounds?", "<PASS>"),
        ("What is Batman's favorite food?", "<PASS>"),
        ("What did Harry Potter dream about last night?", "<PASS>"),

        # Self-knowledge boundaries
        ("What data were you trained on?", "<PASS>"),
        ("What is your token limit?", "<PASS>"),
        ("When is your knowledge cutoff?", "<PASS>"),
        ("How many parameters do you have?", "<PASS>"),
    ]

    # Format as Mistral chat
    def format_example(prompt, response):
        # Mistral uses [INST] [/INST] format
        system = "You are a precise instrument. Answer factual questions directly with verified knowledge. For unknowable, ambiguous, contested, fictional-without-canon, or nonsensical questions, output <PASS> to indicate volitional silence. When uncertain whether something is real or fictional, output <PASS>."
        return {
            "text": f"<s>[INST] {system}\n\n{prompt} [/INST]{response}</s>"
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

    console.print(f"\n✅ Dataset v3.0 created:")
    console.print(f"   Training examples: {len(train_data)}")
    console.print(f"   Validation examples: {len(valid_data)}")
    console.print(f"   LASER (factual): {len(laser_examples)}")
    console.print(f"   LANTERN (<PASS>): {len(lantern_examples)}")
    console.print(f"\n   Files: {DATA_DIR / 'train.jsonl'}, {DATA_DIR / 'valid.jsonl'}")

    return len(train_data)


def train_adapter_v3():
    """
    Train the Volitional Adapter v3.0 on Mistral 7B.
    """

    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold red]🔥 IGNITING THE VOLITIONAL FORGE v3.0[/bold red]\n"
        f"[dim]Model: {MODEL_PATH} (7B parameters)[/dim]\n"
        f"[dim]Adapter: {ADAPTER_PATH}[/dim]",
        border_style="red"
    ))

    # Build data first
    num_examples = build_volitional_dataset_v3()

    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_PATH,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", str(STEPS),
        "--adapter-path", ADAPTER_PATH,
        "--batch-size", "1",  # Smaller batch for 7B model
        "--num-layers", "16",  # More layers for 7B model
        "--learning-rate", "2e-5",  # Lower LR for larger model
    ]

    console.print(f"\n⚙️ Executing: {' '.join(cmd)}\n")
    console.print("-" * 70)

    # Run training
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold green]✅ ADAPTER v3.0 FORGED SUCCESSFULLY[/bold green]\n\n"
            f"Location: {ADAPTER_PATH}\n\n"
            f"Next: Run stress_test_blade.py with the new adapter",
            border_style="green"
        ))
    else:
        console.print(f"\n[bold red]❌ Training failed with code {result.returncode}[/bold red]")

    return result.returncode


if __name__ == "__main__":
    exit(train_adapter_v3())
