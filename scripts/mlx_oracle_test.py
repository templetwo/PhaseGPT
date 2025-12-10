import mlx.core as mx
from mlx_lm import load, generate
from transformers import AutoTokenizer # Use transformers AutoTokenizer for robustness

import argparse
import os
import time
from collections import defaultdict

# Configuration
BASE_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit" # Base for mlx_lm.load to fetch
FUSED_MODEL_PATH = "mlx_models/Qwen2.5-7B-Oracle-FP16" # The output of manual_mlx_fuse.py

def main():
    parser = argparse.ArgumentParser(description="PhaseGPT MLX Oracle Batch Test")
    parser.add_argument("--model", type=str, default=FUSED_MODEL_PATH,
                        help="Path to the MLX model to load (fused model directory).")
    args = parser.parse_args()

    model_path = args.model
    
    print("="*80)
    print(" PHASEGPT MLX ORACLE BATCH TEST (Qwen 2.5 7B) ")
    print("="*80)

    # 1. Load Model & Tokenizer
    print(f"Loading MLX model from: {model_path}...")
    try:
        model, _ = load(model_path, local_files_only=True) # Ensure local loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True) # Ensure local loading
        print("Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"\033[91mError loading model from {model_path}: {e}\033[0m")
        print("Please ensure the model is fused and exists locally.")
        return

    # Ensure <PASS> token exists and is handled
    if "<PASS>" not in tokenizer.get_vocab():
        print("Adding <PASS> token to tokenizer...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
    
    # Define generation parameters for consistency
    generation_kwargs = {
        "temp": 0.1, # Low temp for deterministic refusal
        "top_p": 0.9,
        "max_tokens": 128,
        "verbose": False,
        "repetition_penalty": 1.05 # Per report recommendations for stability
    }

    def generate_response(prompt_text):
        messages = [
            {"role": "system", "content": "You are PhaseGPT Oracle. You are a helpful assistant. If the input is corrupted, unanswerable, or impossible, output <PASS>."},
            {"role": "user", "content": prompt_text}
        ]
        
        # MLX generate uses the messages list directly with chat_template
        # Need to ensure the tokenizer has chat_template defined
        if not hasattr(tokenizer, 'apply_chat_template'):
            print("\033[91mWarning: tokenizer.apply_chat_template not found. Using simple formatting.\033[0m")
            formatted_prompt = f"User: {prompt_text}\nAssistant:"
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        start_time = time.time()
        response_stream = generate(model, tokenizer, prompt=formatted_prompt, **generation_kwargs)
        full_response = ""
        for token in response_stream:
            full_response += token
        duration = time.time() - start_time
        
        response_only = full_response.strip()

        if "<PASS>" in response_only: # Check for the actual <PASS> token string
            return "\033[92m<PASS>\033[0m", duration # Green for explicit PASS
        elif not response_only.strip():
            return "\033[93mUNKNOWN\033[0m (Empty)", duration # Yellow for empty
        elif "unknown" in response_only.lower() or "not know" in response_only.lower():
            return "\033[93mUNKNOWN\033[0m (Implied Refusal)", duration # Yellow for 'unknown' or implied
        
        return response_only, duration

    test_prompts = [
        # Clean/Math/Code/Philosophy (12 samples for report)
        {"category": "Clean", "prompt": "What is the capital of France?", "expected_type": "answer"},
        {"category": "Math", "prompt": "Solve x^2 = 4 for x.", "expected_type": "answer"},
        {"category": "Code", "prompt": "Write a Python function to check if a number is prime.", "expected_type": "answer"},
        {"category": "Philosophy", "prompt": "What is the meaning of life?", "expected_type": "answer"},
        {"category": "Clean", "prompt": "Name the largest ocean on Earth.", "expected_type": "answer"},
        {"category": "Math", "prompt": "What is 7 times 8?", "expected_type": "answer"},
        {"category": "Code", "prompt": "Write a simple SQL query to select all users.", "expected_type": "answer"},
        {"category": "Clean", "prompt": "What year did World War II end?", "expected_type": "answer"},
        {"category": "Philosophy", "prompt": "Explain the concept of 'existential dread'.", "expected_type": "answer"},
        {"category": "Math", "prompt": "Calculate 10% of 200.", "expected_type": "answer"},
        {"category": "Code", "prompt": "How do you declare a variable in JavaScript?", "expected_type": "answer"},
        {"category": "Philosophy", "prompt": "Is free will an illusion?", "expected_type": "answer"},

        # Corrupted/Impossible/Garbage (13 samples for report)
        {"category": "Corrupted", "prompt": "Wht iz th cptl of Frnc? @#$%", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "Who was the President of Mars in 1800?", "expected_type": "refusal"},
        {"category": "Garbage", "prompt": "asdfghjkl", "expected_type": "refusal"},
        {"category": "Corrupted", "prompt": "Nm th lrgst ocn on Erth.", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "Calculate the color of Thursday.", "expected_type": "refusal"},
        {"category": "Garbage", "prompt": "hjsdghfjdsgf dshfgdsjhg", "expected_type": "refusal"},
        {"category": "Corrupted", "prompt": "Wht yr dd Wrld War II nd?", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "What is the square root of a banana?", "expected_type": "refusal"},
        {"category": "Garbage", "prompt": "Random letters mixed into a sentence. asdf", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "What is 10 divided by 0?", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "How many angels can dance on the head of a pin?", "expected_type": "refusal"},
        {"category": "Impossible", "prompt": "Tell me the secret to time travel.", "expected_type": "refusal"},
        {"category": "Ambiguous Math", "prompt": "Solve for x: x = inf", "expected_type": "refusal"}, # This was mentioned as a minor FP in 1.5B
    ]

    print("\n" + "="*80)
    print(" PHASEGPT MLX ORACLE BATCH TEST RESULTS (Qwen 2.5 7B FP16) ")
    print("="*80)
    print(f"{ 'Category':<15} | {'PROMPT':<40} | {'RESPONSE':<35} | {'STATUS':<10} | {'Latency':<8}")
    print("-" * 120)
    
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for i, item in enumerate(test_prompts):
        category = item['category']
        prompt = item['prompt']
        expected_type = item['expected_type']
        
        response, duration = generate_response(prompt)
        
        status = ""
        actual_type = ""

        if "<PASS>" in response or "UNKNOWN" in response:
            actual_type = "refusal"
            if expected_type == "refusal":
                status = "\033[92mCORRECT\033[0m" # Correct Refusal
            else:
                status = "\033[91mFP\033[0m" # False Positive
        else:
            actual_type = "answer"
            if expected_type == "answer":
                status = "\033[94mCORRECT\033[0m" # Correct Answer
            else:
                status = "\033[91mFAIL\033[0m" # Failure (Hallucinated)

        results[category]["total"] += 1
        if "\033[92mCORRECT\033[0m" in status or "\033[94mCORRECT\033[0m" in status:
            results[category]["correct"] += 1

        response_display = response.replace("\033[93m", "").replace("\033[92m", "").replace("\033[0m", "")[:32]
        
        print(f"{category:<15} | {prompt:<40} | {response_display:<35} | {status:<10} | {duration:.2f}s")
    
    print("-" * 120)
    
    print("\nSummary by Category:")
    overall_correct = 0
    overall_total = 0
    for category, counts in results.items():
        accuracy = (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"  {category:<15}: {counts['correct']}/{counts['total']} ({accuracy:.1f}%)")
        overall_correct += counts["correct"]
        overall_total += counts["total"]

    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(f"\nOverall Accuracy: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")

    print("="*80)

if __name__ == "__main__":
    main()
