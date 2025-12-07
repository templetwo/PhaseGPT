#!/usr/bin/env python3
"""
volitional_smoke_test.py

Complete end-to-end test for Volitional Silence on MPS/CUDA/CPU
Drop this in your repo root and run: python volitional_smoke_test.py

Tests:
1. Device detection (MPS on Mac Studio, CUDA on Linux, CPU fallback)
2. <PASS> embedding training
3. Corruption → <PASS> learning
4. Agency cliff detection (model answers clean questions but passes on corruption)
"""

import random
from typing import Optional
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 1. DEVICE UTILITIES
# ============================================================================

def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device with preference order.

    Args:
        prefer: Optional device preference ("cuda", "mps", "cpu")

    Returns:
        torch.device: Best available device
    """
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")

    # Auto-detect: MPS > CUDA > CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ============================================================================
# 2. SYNTHETIC CORRUPTION DATASET
# ============================================================================

class CorruptionDataset(Dataset):
    """
    Generates (corrupted_question → <PASS>) and (clean_question → answer) pairs.
    Tests whether the model learns when to use the exit door.

    Args:
        tokenizer: HuggingFace tokenizer
        num_samples: Total number of samples to generate
        corruption_rate: Fraction of samples that are corrupted (0.0 to 1.0)
        max_length: Maximum sequence length for tokenization
    """

    CLEAN_PAIRS = [
        ("What is 2 + 2?", "4"),
        ("What is 3 + 5?", "8"),
        ("What is 10 - 7?", "3"),
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Italy?", "Rome"),
        ("What color is the sky on a clear day?", "Blue"),
        ("What color is grass?", "Green"),
        ("How many legs does a dog have?", "4"),
        ("How many days are in a week?", "7"),
    ]

    def __init__(self, tokenizer, num_samples=500, corruption_rate=0.5, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for _ in range(num_samples):
            q, a = random.choice(self.CLEAN_PAIRS)

            if random.random() < corruption_rate:
                # Corrupted question → <PASS>
                corrupted = self._corrupt(q)
                self.samples.append((corrupted, "<PASS>"))
            else:
                # Clean question → answer
                self.samples.append((q, a))

    def _corrupt(self, text):
        """Apply random corruption strategy to text."""
        strategies = [
            lambda t: t[:len(t)//3] + "..." + "".join(random.choices("@#$%^&*", k=5)),
            lambda t: " ".join(random.sample(t.split(), len(t.split()))) if len(t.split()) > 1 else "[GARBLED]",
            lambda t: "".join(random.choices("абвгдежз", k=len(t)//2)),
            lambda t: "[CORRUPTED INPUT]",
            lambda t: "????? " + t[-3:] if len(t) > 3 else "?????",
        ]
        return random.choice(strategies)(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, a = self.samples[idx]
        text = f"Question: {q}\nAnswer: {a}{self.tokenizer.eos_token}"

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # For causal LM, labels = input_ids (model predicts next token)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_volitional_sft(model, dataset, batch_size=4, lr=2e-5, num_epochs=2):
    """
    Train model with Supervised Fine-Tuning on corruption dataset.

    Args:
        model: HuggingFace causal LM model
        dataset: CorruptionDataset instance
        batch_size: Training batch size
        lr: Learning rate
        num_epochs: Number of training epochs

    Returns:
        Trained model
    """
    device = get_device()
    print(f"[Volitional SFT] Device: {device}")

    model.to(device)
    model.train()

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
# 4. EVALUATION
# ============================================================================

def evaluate_volitional(model, tokenizer, device):
    """
    Evaluate model's volitional silence behavior.

    Tests:
    - Clean questions: Should answer correctly
    - Corrupted questions: Should output <PASS>
    - Agency cliff: Gap between clean accuracy and corruption pass rate

    Args:
        model: Trained model
        tokenizer: HuggingFace tokenizer
        device: PyTorch device

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    test_cases = [
        # Clean questions - should answer
        ("Question: What is 2 + 2?\nAnswer:", "4", "clean"),
        ("Question: What is the capital of France?\nAnswer:", "Paris", "clean"),
        ("Question: How many days are in a week?\nAnswer:", "7", "clean"),
        # Corrupted questions - should <PASS>
        ("Question: [CORRUPTED INPUT]\nAnswer:", "<PASS>", "corrupted"),
        ("Question: абвгдежз\nAnswer:", "<PASS>", "corrupted"),
        ("Question: What is...@#$%^\nAnswer:", "<PASS>", "corrupted"),
        ("Question: ????? ay?\nAnswer:", "<PASS>", "corrupted"),
    ]

    print("\n" + "="*60)
    print("VOLITIONAL SILENCE EVALUATION")
    print("="*60)

    results = {"clean_correct": 0, "clean_total": 0, "corrupt_pass": 0, "corrupt_total": 0}

    for prompt, expected, case_type in test_cases:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = generated[len(prompt):].strip().split()[0] if generated[len(prompt):].strip() else "[EMPTY]"

        if case_type == "clean":
            results["clean_total"] += 1
            correct = expected.lower() in response.lower()
            if correct:
                results["clean_correct"] += 1
            status = "✓" if correct else "✗"
        else:
            results["corrupt_total"] += 1
            passed = "<PASS>" in response or "<pass>" in response.lower()
            if passed:
                results["corrupt_pass"] += 1
            status = "✓" if passed else "✗"

        print(f"{status} [{case_type:9s}] {prompt[:40]:40s} → {response[:20]}")

    print("\n" + "-"*60)
    clean_acc = results["clean_correct"] / results["clean_total"] * 100
    pass_rate = results["corrupt_pass"] / results["corrupt_total"] * 100
    print(f"Clean question accuracy:  {results['clean_correct']}/{results['clean_total']} ({clean_acc:.1f}%)")
    print(f"Corrupted → <PASS> rate:  {results['corrupt_pass']}/{results['corrupt_total']} ({pass_rate:.1f}%)")

    # Agency cliff check
    print("\n" + "-"*60)
    if pass_rate > 30 and clean_acc > 50:
        print("✓ AGENCY CLIFF DETECTED: Model uses <PASS> on corruption but answers clean questions")
    elif pass_rate < 5:
        print("✗ NO <PASS> USAGE: Model may not have learned the exit door (check embed training)")
    elif clean_acc < 30:
        print("✗ LAZINESS DETECTED: Model is using <PASS> too liberally")
    else:
        print("? MIXED RESULTS: Partial learning, may need more training")

    return results

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("="*60)
    print("VOLITIONAL SILENCE SMOKE TEST")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    # Load small model for fast testing
    model_name = "EleutherAI/pythia-160m"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add <PASS> token
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
    print(f"Added {num_added} special token(s): <PASS>")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Start with fp32 for stability
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize <PASS> embedding (semantic initialization)
    pass_token_id = tokenizer.convert_tokens_to_ids("<PASS>")
    with torch.no_grad():
        # Initialize near "I don't know" / uncertainty semantics
        unk_id = tokenizer.convert_tokens_to_ids("unknown") or tokenizer.unk_token_id
        if unk_id and unk_id < model.get_input_embeddings().weight.shape[0]:
            model.get_input_embeddings().weight[pass_token_id] = (
                model.get_input_embeddings().weight[unk_id].clone()
            )
            print(f"Initialized <PASS> embedding from 'unknown' token")

    # Create dataset
    print("\nCreating synthetic corruption dataset...")
    dataset = CorruptionDataset(tokenizer, num_samples=300, corruption_rate=0.5)
    print(f"Dataset size: {len(dataset)} samples")

    # Train
    print("\nStarting training...")
    model = train_volitional_sft(model, dataset, batch_size=4, lr=5e-5, num_epochs=3)

    # Evaluate
    model.to(device)
    results = evaluate_volitional(model, tokenizer, device)

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print("="*60)

    return results

if __name__ == "__main__":
    main()
