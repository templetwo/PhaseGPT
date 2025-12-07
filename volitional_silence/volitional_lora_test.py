#!/usr/bin/env python3
"""
volitional_lora_test.py

LoRA version of Volitional Silence training with EXPLICIT embed_tokens training.

This version ensures the <PASS> embedding is trainable by including embed_tokens
in the LoRA target modules.

Key difference from vanilla SFT:
- Uses LoRA for parameter efficiency
- **Explicitly includes embed_tokens** in trainable modules
- This ensures <PASS> embedding is actually updated during training

Usage:
    python volitional_lora_test.py
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
    print("Make sure volitional_smoke_test.py is in the same directory")
    sys.exit(1)

# ============================================================================
# LoRA CONFIGURATION WITH EMBED_TOKENS
# ============================================================================

def create_lora_config_with_embeddings():
    """
    Create LoRA config that EXPLICITLY includes embedding layer.

    Critical for Volitional Silence:
    - modules_to_save=["embed_tokens", "lm_head"] ensures these are trainable
    - This allows <PASS> token embedding to be learned
    - Without this, <PASS> embedding stays frozen and model can't use it

    Returns:
        LoraConfig with embedding training enabled
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling factor)
        lora_dropout=0.05,
        target_modules=[
            "q_proj",  # Query projection in attention
            "v_proj",  # Value projection in attention
            "k_proj",  # Key projection (optional but helpful)
            "o_proj",  # Output projection (optional but helpful)
        ],
        # CRITICAL: Include embedding layers in trainable parameters
        modules_to_save=[
            "embed_tokens",  # Input embeddings (where <PASS> lives!)
            "lm_head",       # Output head (also important for <PASS> generation)
        ],
        bias="none",
        inference_mode=False,
    )
    return config

# ============================================================================
# TRAINING LOOP (LoRA version)
# ============================================================================

def train_volitional_lora(model, dataset, batch_size=4, lr=2e-4, num_epochs=3):
    """
    Train model with LoRA on corruption dataset.

    Higher learning rate than vanilla SFT (2e-4 vs 2e-5) because LoRA
    parameters need stronger signal.

    Args:
        model: PEFT model with LoRA adapters
        dataset: CorruptionDataset instance
        batch_size: Training batch size
        lr: Learning rate (higher for LoRA)
        num_epochs: Number of training epochs

    Returns:
        Trained model
    """
    device = get_device()
    print(f"[Volitional LoRA] Device: {device}")

    model.to(device)
    model.train()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Check if embed_tokens is trainable
    embed_trainable = any("embed_tokens" in name for name, p in model.named_parameters() if p.requires_grad)
    if embed_trainable:
        print("✓ embed_tokens is TRAINABLE (good for <PASS> learning)")
    else:
        print("✗ WARNING: embed_tokens is FROZEN - <PASS> may not learn properly!")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Use mixed precision on GPU (Disable for MPS to avoid GradScaler version issues)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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
    print("VOLITIONAL SILENCE - LoRA WITH EMBED TRAINING")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    # Load small model for fast testing
    model_name = "EleutherAI/pythia-160m"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add <PASS> token BEFORE applying LoRA
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
    print(f"Added {num_added} special token(s): <PASS>")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
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

    # Apply LoRA with embed_tokens training
    print("\nApplying LoRA configuration...")
    lora_config = create_lora_config_with_embeddings()
    model = get_peft_model(model, lora_config)

    print("\nLoRA Configuration:")
    print(f"  LoRA rank (r): {lora_config.r}")
    print(f"  LoRA alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    print(f"  Modules to save: {lora_config.modules_to_save}")
    print(f"  ✓ This ensures embed_tokens (where <PASS> lives) is trainable")

    model.print_trainable_parameters()

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

    # For evaluation, we need to merge LoRA weights or use the PEFT model directly
    # PEFT models work fine for inference as-is
    results = evaluate_volitional(model, tokenizer, device)

    print("\n" + "="*60)
    print("LoRA SMOKE TEST COMPLETE")
    print("="*60)
    print("\nKey takeaway:")
    print("If <PASS> usage rate is high, modules_to_save=['embed_tokens'] worked!")
    print("If <PASS> usage rate is low, the embedding may still be frozen somehow.")

    return results

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
