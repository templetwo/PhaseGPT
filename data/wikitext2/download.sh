#!/bin/bash
# Download WikiText-2 dataset for language modeling
# Source: Salesforce Research

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "WikiText-2 dataset download script"
echo ""
echo "Note: This script provides instructions for downloading WikiText-2."
echo "The dataset is typically accessed through Hugging Face datasets or PyTorch text."
echo ""

# Check if using Hugging Face datasets
if python3 -c "import datasets" 2>/dev/null; then
    echo "✓ Hugging Face datasets is installed"
    echo ""
    echo "To load WikiText-2 in your training script:"
    echo ""
    echo "from datasets import load_dataset"
    echo "dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')"
    echo ""
else
    echo "Hugging Face datasets not found. Installing..."
    pip install datasets
fi

# Create a sample loading script
cat > load_wikitext2.py << 'EOF'
"""
Sample script to download and prepare WikiText-2 dataset
"""

from datasets import load_dataset

print("Downloading WikiText-2...")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

print("\nDataset loaded successfully!")
print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print(f"Test examples: {len(dataset['test'])}")

print("\nSample text:")
print(dataset['train'][0]['text'][:500])

print("\nDataset is cached and ready for use!")
print("Access in your training script with:")
print("  dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')")
EOF

echo "Created load_wikitext2.py"
echo ""
echo "Run: python3 load_wikitext2.py"
echo ""
echo "This will download and cache WikiText-2 from Hugging Face."
