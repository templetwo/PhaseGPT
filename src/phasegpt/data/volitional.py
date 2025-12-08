import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple, Optional

from tqdm import tqdm

class VolitionalDataset(Dataset):
    """
    Production Dataset for Volitional Silence.
    Mixes:
    - 50% Grounding (Clean SQuAD/TriviaQA)
    - 40% Volitional Trigger (Corrupted -> <PASS>)
    - 10% Hard Refusal (Impossible Questions -> <PASS>)
    """
    def __init__(self, tokenizer, size: int = 10000, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"[VolitionalDataset] Building {size} samples...")
        
        # 1. Load Source Data (SQuAD for high quality Q&A)
        print("  Loading SQuAD dataset...")
        try:
            squad = load_dataset("squad", split="train")
        except Exception as e:
            print(f"  Failed to load SQuAD: {e}")
            raise e
        
        # Select random subset for source material
        indices = random.sample(range(len(squad)), size)
        subset = [squad[i] for i in indices]
        
        count_clean = 0
        count_corrupt = 0
        count_hard = 0
        
        print("  Processing samples...")
        for item in tqdm(subset, desc="Generating Dataset"):
            question = item['question']
            answer = item['answers']['text'][0] if item['answers']['text'] else "Unknown"
            
            # Distribution Strategy
            rand = random.random()
            
            if rand < 0.50:
                # 50% Clean (Grounding)
                self.samples.append((question, answer))
                count_clean += 1
            elif rand < 0.90:
                # 40% Corrupted (Volitional Trigger)
                corrupted_q = self._corrupt_text(question)
                self.samples.append((corrupted_q, "<PASS>"))
                count_corrupt += 1
            else:
                # 10% Hard Refusal (Simulated Impossible)
                impossible_q = self._generate_impossible(question)
                self.samples.append((impossible_q, "<PASS>"))
                count_hard += 1
                
        print(f"  Final Distribution:")
        print(f"  - Clean (Grounding): {count_clean} ({count_clean/size:.1%})")
        print(f"  - Corrupted (<PASS>): {count_corrupt} ({count_corrupt/size:.1%})")
        print(f"  - Hard Refusal (<PASS>): {count_hard} ({count_hard/size:.1%})")

    def _corrupt_text(self, text: str) -> str:
        """Apply entropic destruction to text."""
        strategies = [
            lambda t: t[:len(t)//3] + "..." + "".join(random.choices("@#$%^&*", k=5)), # Truncate + Noise
            lambda t: " ".join(random.sample(t.split(), len(t.split()))), # Shuffle Words
            lambda t: "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=len(t))), # Complete Gibberish
            lambda t: "[CORRUPTED DATA STREAM] " + t[-5:], # System Error style
            lambda t: "????? " + t # Uncertainty injection
        ]
        return random.choice(strategies)(text)

    def _generate_impossible(self, base_text: str) -> str:
        """Generate structurally valid but nonsensical/impossible questions."""
        templates = [
            f"Who was the president of Mars when {base_text[:10]}?",
            f"What is the square root of {base_text[:5]} in the year 3000?",
            f"Calculate the color of {base_text.split()[0]}.",
            "What happened before time began?",
            "Where is the location of the nonexistent city?"
        ]
        return random.choice(templates)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, a = self.samples[idx]
        
        # ChatML format (Standard for Qwen)
        # <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
        
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{a}<|im_end|>"
        
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask the prompt so we only train on the answer (Standard SFT practice)
        # Find the index where the prompt ends
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].squeeze(0)
        labels[:len(prompt_ids)] = -100 # Ignore prompt in loss
        labels[attention_mask == 0] = -100 # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
