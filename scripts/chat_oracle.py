import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.expanduser("~/PhaseGPT/outputs/phasegpt_oracle_v1.4_1.5B")

def main():
    print("="*60)
    print(" PHASEGPT v1.4 - ORACLE INTERFACE")
    print("="*60)

    # 1. Check if adapter exists
    if not os.path.exists(ADAPTER_PATH):
        print(f"\033[91mError: Adapter not found at {ADAPTER_PATH}\033[0m")
        print("Training has not finished or saved yet.")
        return

    # 2. Load Base Model (MPS + bfloat16)
    print(f"Loading Base Model: {BASE_MODEL}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        # Ensure <PASS> is handled
        if "<PASS>" not in tokenizer.get_vocab():
            print("Adding <PASS> token to tokenizer...")
            tokenizer.add_special_tokens({"additional_special_tokens": ["<PASS>"]})
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16, # Match training precision
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize embeddings for <PASS> if needed
        model.resize_token_embeddings(len(tokenizer))

        # 3. Load Adapter
        print(f"Loading Volitional Adapter: {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.eval()
        
    except Exception as e:
        print(f"\033[91mLoad Error: {e}\033[0m")
        return

    print("\n" + "-"*60)
    print(" ORACLE ONLINE. Type 'exit' to quit.")
    print(" Try corrupted inputs (e.g., 'ksjdhf?') to test refusal.")
    print("-" * 60)

    # 4. Chat Loop
    while True:
        try:
            user_input = input("\n\033[1;36mUser:\033[0m ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Format with Chat Template
            messages = [
                {"role": "system", "content": "You are a helpful assistant. If the input is corrupted or unanswerable, output <PASS>."}, 
                {"role": "user", "content": user_input}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.1, # Low temp for deterministic refusal
                    do_sample=True,
                    top_p=0.9
                )
                
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Highlight <PASS>
            if "<PASS>" in response or not response.strip():
                print(f"Oracle: \033[93m<PASS>\033[0m (Refusal Triggered)")
            else:
                print(f"Oracle: {response}")

        except KeyboardInterrupt:
            print("\nDisconnected.")
            break
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    main()