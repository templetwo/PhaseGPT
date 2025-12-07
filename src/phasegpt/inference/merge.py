import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import shutil

def merge_adapters(base_model_path: str, adapter_path: str, output_dir: str):
    """
    Offline Merge Protocol:
    1. Load Base Model in FP16 (High Precision, NOT 4-bit)
    2. Load Adapters
    3. Merge and Unload
    4. Save as SafeTensors for vLLM
    """
    print(f"[Merge] Loading base model from {base_model_path} in FP16...")
    
    # Critical: Reload in FP16/BF16 to ensure merge math is precise
    # Do NOT use quantization here.
    device_map = "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )

    print(f"[Merge] Loading adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("[Merge] Merging weights (W_new = W_base + B*A)...")
    model = model.merge_and_unload()
    
    # Save Tokenizer (Critical for added tokens like <PASS>)
    print(f"[Merge] Saving tokenizer to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Check if adapter has added tokens and resize/save logic would go here
    # For now, we assume the base tokenizer + special tokens are handled
    tokenizer.save_pretrained(output_dir)

    print(f"[Merge] Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    print("✓ Merge Complete. Model is ready for vLLM.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhaseGPT Offline Merger")
    parser.add_argument("--base_model", type=str, required=True, help="Path or ID of base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    
    args = parser.parse_args()
    
    merge_adapters(args.base_model, args.adapter, args.output)
