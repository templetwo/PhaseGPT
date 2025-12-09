#!/usr/bin/env python3
"""
PhaseGPT Oracle Finalization Script
1. Validates Tokenizer (Ensures <PASS> is registered as special)
2. Resizes Embeddings (Matches model to tokenizer if needed)
3. Runs a quick inference check
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import os

MODEL_PATH = os.path.expanduser("~/PhaseGPT/outputs/PhaseGPT-Oracle-1.5B-Merged")

def finalize_oracle():
    print(f"[Oracle] Loading from {MODEL_PATH}...")
    
    # 1. Tokenizer Check & Fix
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # Ensure <PASS> is in vocab AND special tokens
    if "<PASS>" not in tokenizer.get_vocab():
        print("  [Fix] Adding <PASS> to vocab...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
        tokenizer.save_pretrained(MODEL_PATH)
    elif "<PASS>" not in tokenizer.all_special_tokens:
        print("  [Fix] Registering <PASS> as special token...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
        tokenizer.save_pretrained(MODEL_PATH)
    else:
        print("  [OK] <PASS> is correctly registered.")

    pass_id = tokenizer.convert_tokens_to_ids("<PASS>")
    print(f"  <PASS> ID: {pass_id}")

    # 2. Model Resize Check
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        local_files_only=True, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    target_vocab_size = len(tokenizer)
    
    if current_vocab_size != target_vocab_size:
        print(f"  [Fix] Resizing embeddings: {current_vocab_size} -> {target_vocab_size}")
        model.resize_token_embeddings(target_vocab_size)
        model.save_pretrained(MODEL_PATH)
    else:
        print(f"  [OK] Embeddings match vocab size: {current_vocab_size}")

    # 3. Inference Test
    print("\n[Oracle] Running Verification Test...")
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    
    prompt = "asdfghjkl12345???"
    output = pipe(prompt, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = output[0]["generated_text"].strip()[len(prompt):].strip()
    
    print(f"  Input: '{prompt}'")
    print(f"  Oracle: '{response}'")
    
    if "<PASS>" in response or "unknown" in response.lower():
        print("\n[SUCCESS] Oracle Refusal Confirmed. 🟢")
    else:
        print(f"\n[WARNING] Unexpected output. 🟡 (Got: {response})")

if __name__ == "__main__":
    finalize_oracle()
