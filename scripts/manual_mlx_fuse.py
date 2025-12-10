import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.utils import apply_lora_layers
from mlx_lm.utils import save_model, get_model_path as get_base_model_path, get_tokenizer

import argparse
import os

def manual_mlx_fuse(
    model_path: str,
    adapter_path: str,
    save_path: str,
    # This will dequantize to FP16. If we need 4-bit, we re-quantize after.
):
    print(f"Loading base model from {model_path}...")
    model, tokenizer = load(model_path)
    
    print(f"Applying LoRA adapter from {adapter_path}...")
    model = apply_lora_layers(model, adapter_path) # This applies the LoRA layers to the model
    
    # After apply_lora_layers, the model is still quantized if base was quantized.
    # To get FP16, we need to explicitly convert it.
    
    # Fusion to full precision (FP16/BF16) requires copying to full precision
    # mlx_lm.fuse typically handles this, but since it's broken, we do it manually.
    print("Dequantizing (converting to FP16/BF16) and fusing weights...")
    
    # This loop assumes the model has `_q_module_map` to identify quantized modules.
    # mlx-lm models usually have this.
    for m in model.modules():
        if hasattr(m, "weight") and hasattr(m.weight, "shape") and m.weight.dtype == mx.int8:
            # Assumes it's an int8 quantized linear layer (mlx-lm's 4-bit is often int8 with scale)
            # Convert to float16 explicitly
            m.weight = mx.array(m.weight.astype(mx.float16))
            if hasattr(m, "scales"): # If it's a quantized layer with scales
                 m.scales = mx.array(m.scales.astype(mx.float16))
            if hasattr(m, "biases"):
                 m.biases = mx.array(m.biases.astype(mx.float16))
    
    # After this, the model's weights should be in float16.
    # The `fuse` operation in mlx_lm usually merges A, B into weight.
    # But `apply_lora_layers` already took care of modifying `weight`.
    # So we just save the model.
    
    # The mlx_lm.fuse tool normally has an internal `fuse` function that performs W_new = W_base + (B@A)
    # apply_lora_layers modifies the modules to make this happen.
    # What we're saving here is the full model with the LORA weights applied in full precision.
    
    print(f"Saving fused model to {save_path}...")
    save_model(save_path, model, tokenizer)
    print("Manual fusion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual MLX Adapter Fusion")
    parser.add_argument("--model", type=str, required=True, help="Path to base MLX model (e.g., mlx_models/Qwen2_5-7B-Instruct-4bit)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the trained MLX adapter (e.g., adapters/phasegpt_oracle_7b)")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the fused FP16 model")
    args = parser.parse_args()
    
    manual_mlx_fuse(args.model, args.adapter, args.save_path)
