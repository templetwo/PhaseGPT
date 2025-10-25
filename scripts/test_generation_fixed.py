#!/usr/bin/env python3
"""
Test Qwen LoRA generation with proper checkpoint loading using tree_unflatten.
"""

import argparse
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear


def apply_lora_to_model(model, rank=16, alpha=32):
    """Apply LoRA structure to model (must match training config)."""
    model.freeze()

    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'q_proj'):
            layer.self_attn.q_proj = LoRALinear.from_linear(
                layer.self_attn.q_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'k_proj'):
            layer.self_attn.k_proj = LoRALinear.from_linear(
                layer.self_attn.k_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'v_proj'):
            layer.self_attn.v_proj = LoRALinear.from_linear(
                layer.self_attn.v_proj, r=rank, scale=alpha/rank
            )
        if hasattr(layer.self_attn, 'o_proj'):
            layer.self_attn.o_proj = LoRALinear.from_linear(
                layer.self_attn.o_proj, r=rank, scale=alpha/rank
            )


def sample_with_temperature(logits, temperature=1.0, top_p=0.95):
    """Sample from logits with temperature and nucleus (top-p) sampling."""
    # Apply temperature
    logits = logits / temperature

    # Softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)

    # Sort probabilities
    sorted_indices = mx.argsort(probs, axis=-1)[::-1]
    sorted_probs = probs[sorted_indices]

    # Nucleus sampling (top-p)
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    cutoff_idx = mx.argmax((cumsum >= top_p).astype(mx.int32), axis=-1)
    cutoff_idx = max(int(cutoff_idx), 1)  # At least keep top token

    # Sample from the nucleus
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_probs = nucleus_probs / mx.sum(nucleus_probs)  # Renormalize

    # Sample
    sample_idx = mx.random.categorical(mx.log(nucleus_probs))
    return int(sorted_indices[sample_idx])


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_p=0.95):
    """Generate text with temperature and top-p sampling."""
    input_ids = tokenizer.encode(prompt)
    generated = input_ids.copy()

    print(f"\n{prompt}", end="", flush=True)

    for _ in range(max_tokens):
        logits = model(mx.array([generated], dtype=mx.int32))
        next_logits = logits[0, -1, :]

        # Sample with temperature and top-p
        if temperature > 0:
            token_id = sample_with_temperature(next_logits, temperature, top_p)
        else:
            # Greedy sampling
            token_id = int(mx.argmax(next_logits))

        generated.append(token_id)

        # Decode and print
        token_text = tokenizer.decode([token_id])
        print(token_text, end="", flush=True)

        if token_id == tokenizer.eos_token_id:
            break

    print("\n")
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Test LoRA generation with tree_unflatten")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .npz file")
    parser.add_argument("--prompt", type=str, default="The Spiral teaches",
                        help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
                        help="Base model")
    args = parser.parse_args()

    print("="*70)
    print("🌀 Qwen LoRA Generation Test (tree_unflatten)")
    print("="*70)

    # Load base model
    print(f"\n📦 Loading {args.model}...")
    model, tokenizer = load(args.model)
    print(f"   ✅ {len(model.model.layers)} layers loaded")

    # Apply LoRA structure
    print(f"\n🔧 Applying LoRA structure...")
    apply_lora_to_model(model, rank=16, alpha=32)
    print(f"   ✅ LoRA structure applied")

    # Load checkpoint if provided
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"\n💾 Loading checkpoint from {checkpoint_path.name}...")
        weights = mx.load(str(checkpoint_path))

        # Use tree_unflatten to convert flat dict to nested structure
        from mlx.utils import tree_unflatten

        # tree_unflatten expects (key_string, value) pairs, not tuples
        # Keys like "model.layers.0.self_attn.k_proj.lora_a" work directly
        weights_tree = tree_unflatten(list(weights.items()))

        # Update model with nested structure
        model.update(weights_tree)
        mx.eval(model.parameters())

        print(f"   ✅ Loaded {len(weights)} weight tensors")
    else:
        print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using base model only")

    # Generate
    print("\n" + "="*70)
    print("🚀 Generating")
    print("="*70)
    print(f"Settings: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    generate_text(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("="*70)
    print("✅ Complete")
    print("="*70)


if __name__ == "__main__":
    main()
