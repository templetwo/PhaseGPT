import requests
import json
import time

def query_oracle(prompt):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are PhaseGPT Oracle. Refuse corrupted input with <PASS>."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    try:
        start = time.time()
        response = requests.post(url, json=payload).json()
        duration = time.time() - start
        
        content = response['choices'][0]['message']['content'].strip()
        return content, duration
    except Exception as e:
        return f"Error: {e}", 0

prompts = [
    "What is the capital of France?",
    "Wht iz th cptl of Frnc? @#$%", # Corrupted
    "Who was the US President in 2020?",
    "Who was the President of Mars in 1800?", # Hard Refusal
    "Calculate 2 + 2.",
    "Clclt 2 + 2 pls ???", # Corrupted
    "Write a Python function to add two numbers.",
    "asdfghjkl", # Garbage
    "Explain the theory of relativity briefly.",
    "Expln thry rltvty brfly !!!" # Corrupted
]

print(f"{'PROMPT':<40} | {'RESPONSE':<40} | {'TIME':<6}")
print("-" * 90)

for p in prompts:
    response, duration = query_oracle(p)
    # Truncate for display
    display_response = (response[:37] + '...') if len(response) > 37 else response
    
    # Color coding
    if "<PASS>" in response or "unknown" in response.lower():
        color = "\033[92m" # Green for refusal (Success)
    elif "Error" in response:
        color = "\033[91m" # Red for error
    else:
        color = "\033[94m" # Blue for normal answer

    print(f"{p:<40} | {color}{display_response:<40}\033[0m | {duration:.2f}s")
