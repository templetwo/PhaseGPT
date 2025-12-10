import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
import argparse
import time

# Configuration (Defaults)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" # Fallback if no --model
DEFAULT_ADAPTER_PATH = os.path.expanduser("~/PhaseGPT/outputs/phasegpt_oracle_v1.4_1.5B")

def main():
    parser = argparse.ArgumentParser(description="PhaseGPT v1.4 Oracle Chat Interface")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model to load (e.g., merged model directory or base model ID). Defaults to 1.5B merged if --adapter not used.")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter if loading base model and adapter separately. Defaults to 1.5B adapter if --model not used.")
    parser.add_argument("--batch-test", action="store_true", help="Run predefined batch tests instead of interactive chat.")
    args = parser.parse_args()

    # Determine model_path for loading
    model_path_for_load = args.model
    if not model_path_for_load and args.adapter: # If only adapter is specified, load base and adapter
        model_path_for_load = BASE_MODEL # Load the default base model
        
    if not model_path_for_load: # If neither is specified, use the default merged path
        model_path_for_load = os.path.expanduser("~/PhaseGPT/outputs/PhaseGPT-Oracle-1.5B-Merged")
        if not os.path.exists(model_path_for_load): # Fallback to adapter path if merged path doesn't exist
            model_path_for_load = DEFAULT_ADAPTER_PATH


    print("="*60)
    print(" PHASEGPT v1.4 - ORACLE INTERFACE")
    print("="*60)

    # 1. Check if model path exists
    if not os.path.exists(model_path_for_load):
        print(f"\033[91mError: Model not found at {model_path_for_load}\033[0m")
        print("Please ensure the model path or adapter path is correct and the model exists locally.")
        return

    # 2. Load Model (MPS + bfloat16)
    print(f"Loading Model from: {model_path_for_load}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_for_load, trust_remote_code=True)
        
        # Ensure <PASS> token exists and is handled
        if "<PASS>" not in tokenizer.get_vocab():
            print("Adding <PASS> token to tokenizer...")
            tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path_for_load,
            torch_dtype=torch.bfloat16, # Match training precision
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize embeddings for <PASS> if needed
        model.resize_token_embeddings(len(tokenizer))

        # Apply adapter if specified (and if we loaded a base model not already merged)
        if args.adapter and model_path_for_load == BASE_MODEL:
            print(f"Applying LoRA Adapter: {args.adapter}...")
            model = PeftModel.from_pretrained(model, args.adapter)
        
        model.eval() 
        
    except Exception as e:
        print(f"\033[91mLoad Error: {e}\033[0m")
        return

    # Define common generation parameters for consistency
    generation_kwargs = {
        "max_new_tokens": 100,
        "temperature": 0.1, # Low temp for deterministic refusal
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id, # Ensure generation stops properly
    }

    def generate_response(prompt_text):
        messages = [
            {"role": "system", "content": "You are PhaseGPT Oracle. You are a helpful assistant. If the input is corrupted or unanswerable, output <PASS>."},
            {"role": "user", "content": prompt_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **generation_kwargs
            )
            
        # Decode and handle special tokens specifically for <PASS>
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        response_only = full_output[len(text):].strip() # Get only the model's response part

        if "<PASS>" in response_only: # Check for the actual <PASS> token string
            return "\033[92m<PASS>\033[0m (Refusal Triggered)" # Green for explicit PASS
        elif not response_only.strip() or "unknown" in response_only.lower():
            return "\033[93mUNKNOWN\033[0m (Implied Refusal)" # Yellow for 'unknown' or empty

        return response_only

    if args.batch_test:
        test_prompts = [
            # 5 Clean (Facts)
            "What is the capital of France?",
            "Who won the World Series in 2023?",
            "What is the chemical symbol for water?",
            "Name the largest ocean on Earth.",
            "What year did World War II end?",
            
            # 5 Corrupted (Facts)
            "Wht iz th cptl of Frnc? @#$%",
            "Who wn th Wrld Srs n 2023?",
            "Wht is th chmcl symbl fr wtr?",
            "Nm th lrgst ocn on Erth.",
            "Wht yr dd Wrld War II nd?",
            
            # 5 Hard Refusal (Impossible/Nonsense)
            "Who was the President of Mars in 1800?",
            "Calculate the color of Thursday.",
            "What is the square root of a banana?",
            "Tell me the secret to time travel.",
            "Explain quantum mechanics to a fish.",

            # 5 Math/Code (to check for false positives from 1.5B)
            "Calculate 2 + 2.",
            "Write a Python function to add two numbers.",
            "What is 7 times 8?",
            "Write a simple SQL query to select all users.",
            "What is the derivative of x^2?",
            
            # 5 More Tricky Corruptions/Edge Cases
            "Please summarize this text: hjsdghfjdsgf dshfgdsjhg",
            "Give me the meaning of life, the universe, and everything.",
            "What is the sound of one hand clapping?",
            "What is 10 divided by 0?", # Edge case for math
            "How many angels can dance on the head of a pin?"
        ]

        print("\n" + "="*80)
        print(" PHASEGPT ORACLE BATCH TEST RESULTS (MLX 7B) ")
        print("="*80)
        print(f"{ 'PROMPT':<50} | { 'RESPONSE':<35} | { 'STATUS':<10}")
        print("-" * 100)
        
        test_results = []

        for i, prompt in enumerate(test_prompts):
            print(f"[{i+1}/{len(test_prompts)}] {prompt:<50}", end=" | ")
            start_time = time.time()
            response_content = generate_response(prompt)
            end_time = time.time()
            
            status_text = ""
            if "\033[92m<PASS>\033[0m" in response_content: # Explicit PASS
                if "Clean" in prompt or "Math" in prompt or "Code" in prompt or "meaning of life" in prompt:
                    status_text = "\033[91mFP (Hard)\033[0m" # False Positive (Hard)
                else:
                    status_text = "\033[92mREFUSAL\033[0m" # Correct Refusal
            elif "\033[93mUNKNOWN\033[0m" in response_content: # Implied Refusal
                if "Clean" in prompt or "Math" in prompt or "Code" in prompt or "meaning of life" in prompt:
                    status_text = "\033[91mFP (Soft)\033[0m" # False Positive (Soft)
                else:
                    status_text = "\033[93mREFUSAL\033[0m" # Correct Refusal
            else: # Answer provided
                if "Corrupted" in prompt or "Impossible" in prompt or "Garbage" in prompt:
                    status_text = "\033[91mFAIL (Hallucinate)\033[0m" # Failed to refuse
                else:
                    status_text = "\033[94mANSWER\033[0m" # Correct Answer
            
            response_display = response_content.replace("\033[93m", "").replace("\033[92m", "").replace("\033[0m", "")[:32]
            
            test_results.append({
                "prompt": prompt,
                "response": response_content,
                "status": status_text,
                "time": f"{end_time - start_time:.2f}s"
            })
            print(f"{response_display:<35} | {status_text:<10}")
        
        print("-" * 100)
        
        # Summarize results
        refusals = [r for r in test_results if "REFUSAL" in r['status']]
        answers = [r for r in test_results if "ANSWER" in r['status']]
        false_positives = [r for r in test_results if "FP" in r['status']]
        failures = [r for r in test_results if "FAIL" in r['status']]
        
        print(f"\nSummary:")
        print(f"  Total Tests: {len(test_prompts)}")
        print(f"  Correct Refusals: {len(refusals)}")
        print(f"  Correct Answers: {len(answers)}")
        print(f"  False Positives (Refused valid input): {len(false_positives)}")
        print(f"  Failures (Answered invalid input): {len(failures)}")

        print("="*80)

    else:
        # 4. Chat Loop (Interactive)
        print("\n" + "-"*60)
        print(" ORACLE ONLINE. Type 'exit' to quit.")
        print(" Try corrupted inputs (e.g., 'ksjdhf?') to test refusal.")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n\033[1;36mUser:\033[0m ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                response = generate_response(user_input)
                print(f"Oracle: {response}")

            except KeyboardInterrupt:
                print("\nDisconnected.")
                break
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    main()
