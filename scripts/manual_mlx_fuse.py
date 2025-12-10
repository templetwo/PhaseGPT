import argparse
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import (
    dequantize_model,
    save_model,
    tree_flatten,
    tree_unflatten,
)
# Note: get_model_path from mlx.utils was removed/deprecated. 
# get_base_model_path can be inferred from mlx_lm.utils as well, but not needed here.

def manual_mlx_fuse(model_path: str, adapter_path: str, save_path: str):
    print(f"Loading base model with adapter from {model_path}...")
    # `load` with adapter_path applies LoRA deltas on-the-fly.
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    print("Fusing LoRA layers...")
    # Iterate named_modules(); for LoRA-enabled linears (hasattr(m, 'fuse')), 
    # calls m.fuse(dequantize=True) to merge W += (B @ A) * scale.
    fused_linears = [
        (n, m.fuse(dequantize=True))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    
    if fused_linears:
        # Reconstruct the model with updated fused modules
        model.update_modules(tree_unflatten(fused_linears))
        print(f"Fused {len(fused_linears)} linear layers.")
    
    print("Dequantizing to FP16...")
    # Converts all quantized params (int4/int8) to fp16, popping quantization config.
    model = dequantize_model(model)
    
    print(f"Saving fused FP16 model to {save_path}...")
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
    
    save_model(save_path_obj, model)  
    tokenizer.save_pretrained(save_path_obj)

    # --- CRITICAL FIX: Copy config.json from base model cache ---
    # mlx_lm.load expects config.json and tokenizer_config.json for local loading
    # save_model only saves weights, not config.
    try:
        base_model_cache_path = Path(get_base_model_path(model_path)) # Get MLX cache path of base model
        shutil.copy(base_model_cache_path / "config.json", save_path_obj / "config.json")
        shutil.copy(base_model_cache_path / "tokenizer_config.json", save_path_obj / "tokenizer_config.json")
        print("Copied config and tokenizer config from base model cache.")
    except Exception as e:
        print(f"Warning: Failed to copy config files from base model cache: {e}")
    # --- END CRITICAL FIX ---

    print("Fusion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual MLX LoRA Fusion (Dequantized)")
    parser.add_argument("--model", type=str, required=True, help="Base MLX model path (e.g., mlx-community/Qwen2.5-7B-Instruct-4bit)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the trained MLX adapter (e.g., adapters/phasegpt_oracle_7b)")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the fused FP16 model")
    args = parser.parse_args()
    manual_mlx_fuse(args.model, args.adapter, args.save_path)
