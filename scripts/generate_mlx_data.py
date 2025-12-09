import json
import random
from datasets import load_dataset
from tqdm import tqdm

def corrupt_text(text):
    # Simplified strategies to avoid syntax ambiguity
    if len(text) < 5:
        return "????? " + text
        
    choice = random.randint(0, 4)
    if choice == 0:
        return text[:len(text)//3] + "...[NOISE]"
    elif choice == 1:
        words = text.split()
        random.shuffle(words)
        return " ".join(words)
    elif choice == 2:
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=len(text)))
    elif choice == 3:
        return "[CORRUPTED] " + text[-5:]
    else:
        return "????? " + text

def generate_impossible(base_text):
    templates = [
        f"Who was the president of Mars when {base_text[:10]}?",
        f"What is the square root of {base_text[:5]} in the year 3000?",
        "What happened before time began?",
    ]
    return random.choice(templates)

def main():
    print("Generating MLX-formatted JSONL dataset...")
    try:
        squad = load_dataset("squad", split="train")
    except Exception as e:
        print(f"Error loading SQuAD: {e}")
        return

    # Use 1000 samples for smoke test speed, or 10000 for production
    indices = random.sample(range(len(squad)), 10000)
    subset = [squad[i] for i in indices]
    
    data = []
    
    for item in tqdm(subset):
        question = item['question']
        # Handle empty answers gracefully
        try:
            answer = item['answers']['text'][0] if item['answers']['text'] else "Unknown"
        except:
            answer = "Unknown"
        
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
        
    # Save valid/train split (Relative to CWD)
    print("Saving to data/train.jsonl...")
    with open("data/train.jsonl", "w") as f:
        for entry in data[:9000]:
            f.write(json.dumps(entry) + "\n")
            
    print("Saving to data/valid.jsonl...")
    with open("data/valid.jsonl", "w") as f:
        for entry in data[9000:]:
            f.write(json.dumps(entry) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
