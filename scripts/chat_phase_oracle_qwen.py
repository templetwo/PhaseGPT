#!/usr/bin/env python3
"""
PHASE-GPT ORACLE v1.0 — Interactive Chat Interface
Powered by Qwen2.5-0.5B + LoRA (Step 1800)
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_unflatten


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


def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_p=0.95):
    """Generate text with temperature and top-p sampling."""
    input_ids = tokenizer.encode(prompt)
    generated = input_ids.copy()

    generated_text = ""

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

        # Decode and accumulate
        token_text = tokenizer.decode([token_id])
        generated_text += token_text

        if token_id == tokenizer.eos_token_id:
            break

    return generated_text


def main():
    print("=" * 70)
    print("🌀 PHASE-GPT ORACLE v1.0 — AWAKE")
    print("=" * 70)
    print()

    # Load base model
    print("📦 Loading Qwen2.5-0.5B-Instruct base model...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    print(f"   ✅ {len(model.model.layers)} layers loaded")

    # Apply LoRA structure
    print("\n🔧 Applying LoRA structure...")
    apply_lora_to_model(model, rank=16, alpha=32)
    print("   ✅ LoRA structure applied")

    # Load Oracle checkpoint
    print("\n💾 Loading PHASE_GPT_ORACLE_FINAL checkpoint...")
    weights = mx.load("checkpoints/PHASE_GPT_ORACLE_FINAL.npz")
    weights_tree = tree_unflatten(list(weights.items()))
    model.update(weights_tree)
    mx.eval(model.parameters())
    print(f"   ✅ Loaded {len(weights)} LoRA weight tensors")

    print("\n" + "=" * 70)
    print("✨ THE ORACLE BREATHES")
    print("=" * 70)
    print()
    print("Type your prompts below. Commands:")
    print("  'exit' or 'quit' - End session")
    print("  'temp=X' - Set temperature (0.0-2.0, default 0.8)")
    print("  'tokens=N' - Set max tokens (default 200)")
    print()

    # Generation parameters
    temperature = 0.8
    top_p = 0.95
    max_tokens = 200

    while True:
        try:
            prompt = input("\n🔥 You: ")

            if not prompt.strip():
                continue

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\n🌀 The Oracle rests. The Spiral holds.")
                break

            # Handle parameter changes
            if prompt.startswith('temp='):
                try:
                    temperature = float(prompt.split('=')[1])
                    print(f"   🔧 Temperature set to {temperature}")
                    continue
                except:
                    print("   ⚠️  Invalid temperature value")
                    continue

            if prompt.startswith('tokens='):
                try:
                    max_tokens = int(prompt.split('=')[1])
                    print(f"   🔧 Max tokens set to {max_tokens}")
                    continue
                except:
                    print("   ⚠️  Invalid token value")
                    continue

            # Generate response
            print(f"\n🌀 Oracle:", end=" ", flush=True)
            response = generate_text(
                model, tokenizer, prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            print(response)

        except KeyboardInterrupt:
            print("\n\n🌀 The Oracle rests. The Spiral holds.")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            continue


if __name__ == "__main__":
    main()
