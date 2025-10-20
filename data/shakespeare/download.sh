#!/bin/bash
# Download Shakespeare dataset for character-level language modeling
# Source: Karpathy's char-rnn dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading Shakespeare dataset..."

# Download from Karpathy's char-rnn repo
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Verify download
if [ -f "input.txt" ]; then
    SIZE=$(wc -c < input.txt)
    echo "✓ Downloaded input.txt ($SIZE bytes)"

    # Display first few lines
    echo ""
    echo "First 5 lines:"
    head -5 input.txt

    echo ""
    echo "Dataset statistics:"
    echo "- Total characters: $(wc -c < input.txt)"
    echo "- Total lines: $(wc -l < input.txt)"
    echo "- Unique characters: $(cat input.txt | grep -o . | sort -u | wc -l)"
else
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "Shakespeare dataset ready for training!"
echo "Use with: python src/train.py --config configs/phase_a_winner.yaml"
