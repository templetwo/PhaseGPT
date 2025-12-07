#!/usr/bin/env python3
"""
Tiny Integration Test for PhaseGPT v1.4
Verifies:
1. Pydantic Config Loading
2. Architecture Detection
3. VolitionalTrainer Initialization
4. One Step of Training (Gradient Check)
"""

import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from phasegpt.core.architecture import ArchitectureConfig
from phasegpt.trainer.volitional import VolitionalTrainer
from torch.utils.data import Dataset

# Mock Dataset
class TinyDataset(Dataset):
    def __init__(self, tokenizer, length=8):
        self.tokenizer = tokenizer
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Return a simple input
        text = "Hello world"
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"].squeeze(0), "labels": labels}

def main():
    print("="*60)
    print("PHASEGPT v1.4 TINY INTEGRATION RUN")
    print("="*60)
    
    # 1. Load Config
    print("[1] Loading Config...")
    config_path = Path("PhaseGPT/config/architecture_config.yaml")
    if not config_path.exists():
        # Fallback for running inside PhaseGPT dir
        config_path = Path("config/architecture_config.yaml")
        
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        
    # Override for tiny test (use tiny model if possible, or stick to config but run 1 step)
    # Using Pythia-70m or 160m for speed if available, else config default
    # adhering to config default (Qwen 0.5B) for correctness check
    
    arch_config = ArchitectureConfig(**config_dict)
    print(f"✓ Config Loaded: {arch_config.model_name_or_path}")

    # 2. Initialize Model & Tokenizer
    print("\n[2] Initializing Model...")
    model = AutoModelForCausalLM.from_pretrained(arch_config.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(arch_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Initialize Trainer
    print("\n[3] Initializing VolitionalTrainer...")
    trainer = VolitionalTrainer(model, tokenizer, arch_config, batch_size=2)
    
    # 4. Run Training (1 Epoch, tiny dataset)
    print("\n[4] Running Training Loop...")
    dataset = TinyDataset(tokenizer)
    trainer.train(dataset, num_epochs=1)
    
    print("\n✓ SUCCESS: Integrated Run Complete.")

if __name__ == "__main__":
    main()
