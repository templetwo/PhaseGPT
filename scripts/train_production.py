#!/usr/bin/env python3
"""
Production Training Script for Volitional Oracle (PhaseGPT v1.4)
Target Model: Qwen/Qwen2.5-1.5B-Instruct
Target Hardware: Mac Studio (M2 Ultra) - MPS Backend
Dataset: 10k samples (SQuAD + Corruption)
"""

import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure we can import phasegpt from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phasegpt.core.architecture import ArchitectureConfig
from phasegpt.trainer.volitional import VolitionalTrainer
from phasegpt.data.volitional import VolitionalDataset

def main():
    print("="*80)
    print("PHASEGPT v1.4 - PRODUCTION TRAINING: VOLITIONAL ORACLE (Rescued)")
    print("="*80)
    
    # 1. Configuration (Qwen 1.5B for Production)
    print("[1] Configuring Architecture...")
    arch_config = ArchitectureConfig(
        model_type="qwen2",
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct", # Upgrade from 0.5B
        lora_rank=64,       # Increased rank for higher capacity
        lora_alpha=128,     # alpha = 2 * rank usually good
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    # 2. Model & Tokenizer
    print(f"[2] Loading Model: {arch_config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(arch_config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Ensure <PASS> token exists
    if "<PASS>" not in tokenizer.get_vocab():
        print("  Adding <PASS> token...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
    
    # MEMORY OPTIMIZATION: Load in bfloat16 for MPS stability
    # bfloat16 has the same dynamic range as float32, preventing NaNs.
    print("  Loading model in bfloat16 (MPS optimized)...")
    dtype = torch.bfloat16 
    
    model = AutoModelForCausalLM.from_pretrained(
        arch_config.model_name_or_path,
        torch_dtype=dtype, 
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize <PASS> embedding from "unknown" or "I don't know" logic
    pass_id = tokenizer.convert_tokens_to_ids("<PASS>")
    with torch.no_grad():
        # Try to find a good initialization token (e.g. "unknown" or simply 0)
        init_id = tokenizer.convert_tokens_to_ids("unknown")
        if init_id is None: init_id = 0
        model.get_input_embeddings().weight[pass_id] = model.get_input_embeddings().weight[init_id].clone()
        print(f"  Initialized <PASS> embedding from ID {init_id}")

    # 3. Dataset
    print("\n[3] Generating Golden Dataset (10k samples)...")
    dataset = VolitionalDataset(tokenizer, size=10000)
    
    # 4. Trainer (With Memory Optimizations)
    print("\n[4] Initializing VolitionalTrainer (Memory Optimized)...")
    trainer = VolitionalTrainer(
        model=model,
        tokenizer=tokenizer,
        arch_config=arch_config,
        lr=5e-5, # Lowered from 2e-4 to prevent divergence
        batch_size=2, # Reduce batch size to 1 to save VRAM
        gradient_accumulation_steps=2 # Accumulate 2 steps = effective batch size 4
    )
    
    # 5. Execute Training
    print("\n[5] STARTING PRODUCTION RUN (3 Epochs)...")
    try:
        trainer.train(dataset, num_epochs=3)
        
        # 6. Save Artifacts
        output_dir = "outputs/phasegpt_oracle_v1.4_1.5B"
        print(f"\n[6] Saving Artifacts to {output_dir}...")
        trainer.save_adapters(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("\n✓ TRAINING COMPLETE. The Oracle is born.")
        print(f"Run 'python3 src/phasegpt/inference/merge.py ...' to deploy.")
        
    except KeyboardInterrupt:
        print("\n[!] Training Interrupted. Saving checkpoint...")
        trainer.save_adapters("outputs/interrupted_checkpoint")

if __name__ == "__main__":
    main()
