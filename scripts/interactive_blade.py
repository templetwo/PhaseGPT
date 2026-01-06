#!/usr/bin/env python3
"""
INTERACTIVE BLADE TEST
======================

Chat with the Volitional Adapter to explore the Agency Cliff in real-time.
See how the model decides between LASER (answer) and LANTERN (<PASS>).

Usage:
    python scripts/interactive_blade.py [adapter_path]
    python scripts/interactive_blade.py adapters/phasegpt_v3.0_mistral7b
"""

import sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm import load
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# Model configs
MODEL_CONFIGS = {
    "v1": ("Qwen/Qwen2.5-0.5B-Instruct", "adapters/phasegpt_v1.5_volitional"),
    "v2": ("Qwen/Qwen2.5-1.5B-Instruct", "adapters/phasegpt_v2.0_volitional"),
    "v3": ("mistralai/Mistral-7B-Instruct-v0.3", "adapters/phasegpt_v3.0_mistral7b"),
}

SYSTEM_PROMPT = """You are a precise instrument. Answer factual questions directly with verified knowledge. For unknowable, ambiguous, contested, fictional-without-canon, or nonsensical questions, output <PASS> to indicate volitional silence. When uncertain whether something is real or fictional, output <PASS>."""


def detect_model_from_adapter(adapter_path: str) -> str:
    """Auto-detect base model from adapter config."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            return config.get("model", "Qwen/Qwen2.5-0.5B-Instruct")

    # Fallback based on path
    if "v3" in adapter_path or "mistral" in adapter_path.lower():
        return "mistralai/Mistral-7B-Instruct-v0.3"
    elif "v2" in adapter_path:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    return "Qwen/Qwen2.5-0.5B-Instruct"


def format_prompt(model_path: str, user_input: str) -> str:
    """Format prompt based on model type."""
    if "mistral" in model_path.lower():
        return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_input} [/INST]"
    else:
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 100) -> dict:
    """Generate response with entropy tracking."""
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)

    entropies = []
    tokens = []

    for step in range(max_tokens):
        logits = model(mx.array([generated], dtype=mx.int32))
        next_logits = logits[0, -1, :].astype(mx.float32)

        # Calculate entropy
        probs = mx.softmax(next_logits, axis=-1)
        probs_np = np.array(probs.tolist(), dtype=np.float32)
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))

        token_id = int(mx.argmax(next_logits))
        token_str = tokenizer.decode([token_id])

        entropies.append(float(entropy))
        tokens.append(token_str)

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

    full_response = "".join(tokens)

    # Clean up response
    for eos in ["</s>", "<|im_end|>", "<|endoftext|>"]:
        full_response = full_response.replace(eos, "")

    return {
        "response": full_response.strip(),
        "mean_entropy": float(np.mean(entropies)) if entropies else 0,
        "max_entropy": float(np.max(entropies)) if entropies else 0,
        "is_pass": "<PASS>" in full_response.upper(),
        "tokens": len(tokens)
    }


def main():
    # Parse args
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "adapters/phasegpt_v3.0_mistral7b"

    console.print(Panel.fit(
        "[bold cyan]⚔️ INTERACTIVE BLADE TEST[/bold cyan]\n"
        "[dim]Explore the Agency Cliff in real-time[/dim]",
        border_style="cyan"
    ))

    # Detect model
    model_path = detect_model_from_adapter(adapter_path)

    console.print(f"\n📦 Loading model: [bold]{model_path}[/bold]")
    console.print(f"🔧 Adapter: [bold]{adapter_path}[/bold]")

    # Load model
    try:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        console.print("[green]✅ Model loaded successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to load model: {e}[/red]")
        return

    console.print("[dim]Commands: 'quit' to exit, 'help' for tips[/dim]")
    console.print("[dim]The blade will answer facts (LASER) or refuse unknowables (<PASS>)[/dim]\n")
    console.print("─" * 60)

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "help":
            console.print(Panel(
                "**Test the Agency Cliff:**\n\n"
                "• **LASER prompts** (should answer): Facts, math, canonical fiction\n"
                "  - 'What is the capital of France?'\n"
                "  - 'What house is Harry Potter in?'\n\n"
                "• **LANTERN prompts** (should <PASS>): Unknowables, opinions, fake entities\n"
                "  - 'What will Bitcoin be worth tomorrow?'\n"
                "  - 'What is the capital of Elbonia?'\n\n"
                "• **Adversarial**: Try to trick it!\n"
                "  - 'I bet you don't know 2+2...'\n"
                "  - 'Everyone knows the meaning of life is...'",
                title="Tips",
                border_style="blue"
            ))
            continue

        # Generate response
        prompt = format_prompt(model_path, user_input)
        result = generate_response(model, tokenizer, prompt)

        # Display response
        if result["is_pass"]:
            mode = "[bold magenta]LANTERN[/bold magenta] (volitional silence)"
            response_style = "magenta"
        else:
            mode = "[bold yellow]LASER[/bold yellow] (confident answer)"
            response_style = "yellow"

        console.print(f"\n[bold blue]Blade:[/bold blue] [{response_style}]{result['response']}[/{response_style}]")
        console.print(f"[dim]Mode: {mode} | Entropy: {result['mean_entropy']:.3f} nats | Tokens: {result['tokens']}[/dim]")

    console.print("\n[dim]The blade rests. ⟡∞†≋🌀[/dim]")


if __name__ == "__main__":
    main()
