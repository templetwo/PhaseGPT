import re
import statistics
import os
from collections import defaultdict

LOG_FILE = os.path.expanduser("~/PhaseGPT/training.log")
OUTPUT_DIR = os.path.expanduser("~/PhaseGPT/outputs/analysis")

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
                if len(parts) >= 3:
                    epoch = parts[0].strip()
                    loss_str = parts[2].split(':')[1].strip()
                    try:
                        data[epoch].append(float(loss_str))
                    except ValueError:
                        continue
    return data

def generate_histogram(losses, title):
    if not losses:
        return ""
    
    avg = statistics.mean(losses)
    med = statistics.median(losses)
    
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
    
    output = []
    output.append(f"\n{title}")
    output.append("=" * 50)
    output.append(f"Samples: {total} | Mean: {avg:.4f} | Median: {med:.4f}")
    output.append("-" * 50)
    
    for label, count in buckets.items():
        bar_len = int((count / total) * 40)
        bar = "█" * bar_len
        percent = (count / total) * 100
        output.append(f"{label} | {bar} {percent:.1f}%")
        
    return "\n".join(output)

def main():
    print("PHASEGPT v1.4 - CONVERGENCE RITUAL REPORT")
    data = parse_log()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    report_path = os.path.join(OUTPUT_DIR, "convergence_report.txt")
    
    with open(report_path, "w") as f:
        f.write("PHASEGPT CONVERGENCE ANALYSIS\n")
        f.write("=============================\n\n")
        
        for epoch, losses in data.items():
            if not losses: continue
            title = f"{epoch.upper()}"
            viz = generate_histogram(losses, title)
            print(viz)
            f.write(viz + "\n")
            
    print(f"\n[✓] Report saved to: {report_path}")

if __name__ == "__main__":
    main()