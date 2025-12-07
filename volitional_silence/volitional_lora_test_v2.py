#!/usr/bin/env python3
"""
volitional_lora_test_v2.py

Architecture-aware LoRA configuration for Volitional Silence.

This version AUTO-DETECTS whether the base model uses:
- Qwen/Llama architecture (embed_tokens + lm_head)
- Pythia/GPT-NeoX architecture (embed_in + embed_out)

Based on technical analysis: "Architectural Adaptation of LoRA Configurations
for Special Token Embedding Training"

Key insight: modules_to_save MUST match the actual module names in the
architecture, or the embedding layer remains frozen and <PASS> cannot be learned.

Usage:
    python volitional_lora_test_v2.py
"""

import random
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Reuse utilities from smoke test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from volitional_smoke_test import get_device, CorruptionDataset, evaluate_volitional
except ImportError:
    print("Error: Could not import from volitional_smoke_test.py")
    sys.exit(1)

# ============================================================================
# ARCHITECTURE DETECTION
# ============================================================================

def detect_architecture(model) -> dict:
    """
    Detect model architecture and return appropriate module names.

    Architecture families:
    1. Qwen/Llama lineage: Uses 'embed_tokens' and 'lm_head'
    2. Pythia/GPT-NeoX lineage: Uses 'embed_in' and 'embed_out'

    Returns:
        dict with keys:
            - architecture: "qwen" or "pythia" or "unknown"
            - embed_input: str name of input embedding module
            - embed_output: str name of output projection module
            - target_modules: list of attention/MLP modules to target

    Based on: "Architectural Adaptation of LoRA Configurations for
    Special Token Embedding Training: A Comparative Analysis of
    Qwen2.5 and Pythia Architectures"
    """
    model_type = model.config.model_type.lower()

    # Qwen/Llama family detection
    if any(x in model_type for x in ['qwen', 'llama', 'mistral', 'phi']):
        return {
            "architecture": "qwen_llama",
            "embed_input": "embed_tokens",
            "embed_output": "lm_head",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            "root_module": "model"
        }

    # Pythia/GPT-NeoX family detection
    elif any(x in model_type for x in ['gpt_neox', 'pythia']):
        return {
            "architecture": "pythia_neox",
            "embed_input": "embed_in",
            "embed_output": "embed_out",
            "target_modules": ["query_key_value", "dense",
                             "dense_h_to_4h", "dense_4h_to_h"],
            "root_module": "gpt_neox"
        }

    # Unknown architecture - try to detect from model structure
    else:
        # Introspect the model to find embedding layer
        has_embed_tokens = hasattr(model, 'model') and hasattr(model.model, 'embed_tokens')
        has_embed_in = hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in')

        if has_embed_tokens:
            print(f"⚠️  Unknown model_type '{model_type}', but found 'embed_tokens'")
            print("    Assuming Qwen/Llama architecture")
            return {
                "architecture": "detected_qwen_llama",
                "embed_input": "embed_tokens",
                "embed_output": "lm_head",
                "target_modules": ["q_proj", "v_proj"],  # Conservative
                "root_module": "model"
            }
        elif has_embed_in:
            print(f"⚠️  Unknown model_type '{model_type}', but found 'embed_in'")
            print("    Assuming Pythia/NeoX architecture")
            return {
                "architecture": "detected_pythia_neox",
                "embed_input": "embed_in",
                "embed_output": "embed_out",
                "target_modules": ["query_key_value"],  # Conservative
                "root_module": "gpt_neox"
            }
        else:
            raise ValueError(
                f"❌ Unable to detect architecture for model_type: {model_type}\n"
                f"   Model does not have 'embed_tokens' or 'embed_in'.\n"
                f"   Please manually specify modules_to_save in LoraConfig."
            )

def verify_modules_exist(model, module_names: list) -> dict:
    """
    Verify that specified modules actually exist in the model.

    Returns:
        dict: {module_name: exists (bool)}
    """
    results = {}
    for module_name in module_names:
        # Check if module exists by trying to get it
        found = False
        for name, _ in model.named_modules():
            if module_name in name:
                found = True
                break
        results[module_name] = found
    return results

# ============================================================================
# ARCHITECTURE-AWARE LORA CONFIGURATION
# ============================================================================

def create_architecture_aware_lora_config(model):
    """
    Create LoRA config that adapts to the model's architecture.

    Critical for Volitional Silence:
    - Input embedding (embed_tokens or embed_in) must be trainable
    - Output projection (lm_head or embed_out) must be trainable
    - Both are required due to untied embeddings in modern architectures

    Args:
        model: HuggingFace model instance

    Returns:
        LoraConfig with correct modules_to_save for the architecture
    """
    arch_info = detect_architecture(model)

    print("\n" + "="*60)
    print("ARCHITECTURE DETECTION")
    print("="*60)
    print(f"Model type: {model.config.model_type}")
    print(f"Detected architecture: {arch_info['architecture']}")
    print(f"Root module: {arch_info['root_module']}")
    print(f"Input embedding: {arch_info['embed_input']}")
    print(f"Output projection: {arch_info['embed_output']}")
    print(f"Target modules: {arch_info['target_modules']}")

    # Verify modules exist
    modules_to_check = [arch_info['embed_input'], arch_info['embed_output']]
    verification = verify_modules_exist(model, modules_to_check)

    print("\nModule Verification:")
    for module_name, exists in verification.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {module_name}: {'Found' if exists else 'NOT FOUND'}")

    if not all(verification.values()):
        raise ValueError(
            f"❌ Critical modules not found in model!\n"
            f"   This indicates architecture detection failed.\n"
            f"   Verification: {verification}"
        )

    # Create LoRA config with architecture-specific settings
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=arch_info['target_modules'],

        # CRITICAL: Use architecture-specific module names
        # For Qwen/Llama: ["embed_tokens", "lm_head"]
        # For Pythia/NeoX: ["embed_in", "embed_out"]
        modules_to_save=[
            arch_info['embed_input'],
            arch_info['embed_output']
        ],

        bias="none",
        inference_mode=False,
    )

    print("\n" + "="*60)
    print("LORA CONFIGURATION")
    print("="*60)
    print(f"LoRA rank (r): {config.r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Target modules: {config.target_modules}")
    print(f"Modules to save (FULL FINE-TUNING):")
    for module in config.modules_to_save:
        print(f"  • {module}")
    print("\n✓ Configuration ensures <PASS> embedding is trainable")

    return config, arch_info

# ============================================================================
# TRAINING LOOP (Architecture-aware)
# ============================================================================

def train_volitional_lora(model, dataset, batch_size=4, lr=2e-4, num_epochs=3):
    """Train with LoRA on corruption dataset."""
    device = get_device()
    print(f"\n[Volitional LoRA] Device: {device}")

    model.to(device)
    model.train()

    # Print trainable parameters
    model.print_trainable_parameters()

    # Verify embedding trainability
    print("\nEmbedding Layer Trainability Check:")
    embedding_trainable = False
    for name, param in model.named_parameters():
        if "embed" in name.lower() and param.requires_grad:
            print(f"  ✓ {name}: TRAINABLE (shape: {param.shape})")
            embedding_trainable = True

    if not embedding_trainable:
        print("  ✗ WARNING: No embedding layers are trainable!")
        print("    <PASS> token may not learn properly.")
        print("    This indicates modules_to_save failed to match the layers.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Only use AMP on CUDA, not MPS (GradScaler compatibility issues)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type="cuda") if use_amp else None

    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            try:
                if use_amp:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(**batch)
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                if step % 20 == 0:
                    print(f"  epoch {epoch+1} step {step:3d} loss {loss.item():.4f}")

            except RuntimeError as e:
                if "nan" in str(e).lower() or torch.isnan(loss):
                    print(f"  [WARN] NaN at epoch {epoch+1} step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue
                raise

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average loss: {avg_loss:.4f}")

    return model

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("VOLITIONAL SILENCE - ARCHITECTURE-AWARE LoRA")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    # Test with multiple architectures
    # You can change this to test different models
    model_name = "EleutherAI/pythia-160m"  # Pythia (GPT-NeoX)
    # model_name = "Qwen/Qwen2.5-0.5B"      # Qwen (Llama-lineage)

    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add <PASS> token BEFORE applying LoRA
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
    print(f"Added {num_added} special token(s): <PASS>")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # Resize embeddings BEFORE LoRA wrapping
    model.resize_token_embeddings(len(tokenizer))

    # Initialize <PASS> embedding
    pass_token_id = tokenizer.convert_tokens_to_ids("<PASS>")
    with torch.no_grad():
        unk_id = tokenizer.convert_tokens_to_ids("unknown") or tokenizer.unk_token_id
        if unk_id and unk_id < model.get_input_embeddings().weight.shape[0]:
            model.get_input_embeddings().weight[pass_token_id] = (
                model.get_input_embeddings().weight[unk_id].clone()
            )
            print(f"Initialized <PASS> embedding from 'unknown' token")

    # Apply ARCHITECTURE-AWARE LoRA configuration
    lora_config, arch_info = create_architecture_aware_lora_config(model)
    model = get_peft_model(model, lora_config)

    # Create dataset
    print("\nCreating synthetic corruption dataset...")
    dataset = CorruptionDataset(tokenizer, num_samples=300, corruption_rate=0.5)
    print(f"Dataset size: {len(dataset)} samples")

    # Train with LoRA
    print("\nStarting LoRA training...")
    model = train_volitional_lora(model, dataset, batch_size=4, lr=2e-4, num_epochs=3)

    # Evaluate
    print("\nEvaluating...")
    model.to(device)
    results = evaluate_volitional(model, tokenizer, device)

    print("\n" + "="*60)
    print("ARCHITECTURE-AWARE LoRA TEST COMPLETE")
    print("="*60)
    print(f"\nArchitecture: {arch_info['architecture']}")
    print(f"Embeddings trained: {arch_info['embed_input']}, {arch_info['embed_output']}")
    print("\nKey takeaway:")
    print("If <PASS> usage rate is high, architecture detection worked!")
    print(f"The correct modules ({arch_info['embed_input']}, {arch_info['embed_output']}) were unfrozen.")

    return results, arch_info

if __name__ == "__main__":
    # Check PEFT is installed
    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        print("ERROR: PEFT not installed. Install with:")
        print("  pip install peft")
        sys.exit(1)

    main()
