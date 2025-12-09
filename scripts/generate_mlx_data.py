import json
import random
from datasets import load_dataset
from tqdm import tqdm

def corrupt_text(text):
    strategies = [
        lambda t: t[:len(t)//3] + "..." + "".join(random.choices("@#$%^&*", k=5)), 
        lambda t: " ".join(random.sample(t.split(), len(t.split()))),
        lambda t: "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=len(t))),
        lambda t: "[CORRUPTED] " + t[-5:]
        lambda t: "????? " + t
    ]
    return random.choice(strategies)(text)

def generate_impossible(base_text):
    templates = [
        f"Who was the president of Mars when {base_text[:10]}?",
        f"What is the square root of {base_text[:5]} in the year 3000?",
        "What happened before time began?",
    ]
    return random.choice(templates)

def main():
    print("Generating MLX-formatted JSONL dataset...")
    squad = load_dataset("squad", split="train")
    indices = random.sample(range(len(squad)), 10000)
    subset = [squad[i] for i in indices]
    
    data = []
    
    for item in tqdm(subset):
        question = item['question']
        answer = item['answers']['text'][0] if item['answers']['text'] else "Unknown"
        
        rand = random.random()
        
        if rand < 0.50:
            # Clean
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        elif rand < 0.90:
            # Corrupted
            messages = [
                {"role": "user", "content": corrupt_text(question)},
                {"role": "assistant", "content": "<PASS>"}
            ]
        else:
            # Impossible
            messages = [
                {"role": "user", "content": generate_impossible(question)},
                {"role": "assistant", "content": "<PASS>"}
            ]
            
        data.append({"messages": messages})
        
    # Save valid/train split
    with open("PhaseGPT/data/train.jsonl", "w") as f:
        for entry in data[:9000]:
            f.write(json.dumps(entry) + "\n")
            
    with open("PhaseGPT/data/valid.jsonl", "w") as f:
        for entry in data[9000:]:
            f.write(json.dumps(entry) + "\n")
            
    print("Done. Saved to PhaseGPT/data/train.jsonl and valid.jsonl")

if __name__ == "__main__":
    main()
