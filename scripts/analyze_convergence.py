import re
import statistics
import os
from collections import defaultdict

LOG_FILE = os.path.expanduser("~/PhaseGPT/training.log")

def parse_log():
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return {}

    data = defaultdict(list)
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if "Loss:" in line:
                # Epoch 1 | Step 4999 | Loss: 1.6258
                parts = line.split('|')
                epoch = parts[0].strip()
                loss_str = parts[2].split(':')[1].strip()
                data[epoch].append(float(loss_str))
    return data

def print_histogram(losses, title):
    if not losses:
        return
    
    avg = statistics.mean(losses)
    med = statistics.median(losses)
    
    # Buckets: <0.1 (Perfect), 0.1-1.0 (Good), 1.0-2.0 (Bad), >2.0 (Terrible/Hallucination)
    buckets = {
        "Perfect (<0.1) ": 0,
        "Okay (0.1-1.0)  ": 0,
        "Rough (1.0-2.0) ": 0,
        "Fail (>2.0)     ": 0
    }
    
    for l in losses:
        if l < 0.1: buckets["Perfect (<0.1) "] += 1
        elif l < 1.0: buckets["Okay (0.1-1.0)  "] += 1
        elif l < 2.0: buckets["Rough (1.0-2.0) "] += 1
        else: buckets["Fail (>2.0)     "] += 1
        
    total = len(losses)
    
    print(f"\n{title}")
    print("=" * 50)
    print(f"Samples: {total} | Mean: {avg:.4f} | Median: {med:.4f}")
    print("-" * 50)
    
    max_val = max(buckets.values())
    
    for label, count in buckets.items():
        bar_len = int((count / total) * 40)
        bar = "█" * bar_len
        percent = (count / total) * 100
        print(f"{label} | {bar} {percent:.1f}%")

def main():
    print("PHASEGPT v1.4 - CONVERGENCE RITUAL REPORT")
    data = parse_log()
    
    if "Epoch 1" in data:
        print_histogram(data["Epoch 1"], "EPOCH 1: THE STRUGGLE (Binary Distribution)")
        
    if "Epoch 2" in data:
        print_histogram(data["Epoch 2"], "EPOCH 2: THE AWAKENING (Collapse to Silence)")

if __name__ == "__main__":
    main()
