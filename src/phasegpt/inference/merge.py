import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import shutil

def merge_adapters(base_model_path: str, adapter_path: str, output_dir: str):
    """
    Offline Merge Protocol:
    1. Load Tokenizer (from adapter if available, to get correct vocab size)
    2. Load Base Model in FP16
    3. Resize Base Model Embeddings (Critical for Qwen padding mismatch)
    4. Load Adapters
    5. Merge and Unload
    6. Save
    """
    print(f"[Merge] Loading tokenizer from {adapter_path}...")
    try:
        # Try loading from adapter first (contains <PASS>)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except:
        print(f"[Merge] Adapter tokenizer not found. Loading from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    vocab_size = len(tokenizer)
    print(f"[Merge] Target Vocab Size: {vocab_size}")

    print(f"[Merge] Loading base model from {base_model_path} in FP16...")
    device_map = "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )

    # CRITICAL FIX: Resize embeddings BEFORE loading adapter to match trained dimension
    # This handles the Qwen padding mismatch (151936 vs 151666)
    print(f"[Merge] Resizing embeddings to {vocab_size}...")
    base_model.resize_token_embeddings(vocab_size)

    print(f"[Merge] Loading adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("[Merge] Merging weights (W_new = W_base + B*A)...")
    model = model.merge_and_unload()
    
    print(f"[Merge] Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    print(f"[Merge] Saving tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Merge Complete. Model is ready for vLLM.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhaseGPT Offline Merger")
    parser.add_argument("--base_model", type=str, required=True, help="Path or ID of base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    
    args = parser.parse_args()
    
    merge_adapters(args.base_model, args.adapter, args.output)