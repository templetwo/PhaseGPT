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
from mlx.utils import get_model_path as get_base_model_path  # If needed for HF paths

def manual_mlx_fuse(model_path: str, adapter_path: str, save_path: str):
    print(f"Loading base model with adapter from {model_path}...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    print("Fusing LoRA layers...")
    fused_linears = [
        (n, m.fuse(dequantize=True))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
        print(f"Fused {len(fused_linears)} linear layers.")
    
    print("Dequantizing to FP16...")
    model = dequantize_model(model)
    
    print(f"Saving fused FP16 model to {save_path}...")
    save_path_obj = Path(save_path)
    save_model(save_path_obj, model, tokenizer)  # Assumes save_model handles config
    print("Fusion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual MLX LoRA Fusion (Dequantized)")
    parser.add_argument("--model", type=str, required=True, help="Base MLX model path")
    parser.add_argument("--adapter", type=str, required=True, help="Adapter path")
    parser.add_argument("--save-path", type=str, required=True, help="Output path")
    args = parser.parse_args()
    manual_mlx_fuse(args.model, args.adapter, args.save_path)