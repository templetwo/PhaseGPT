#!/usr/bin/env bash
#
# Find the real PhaseGPT Spiral corpus checkpoint
#
set -euo pipefail

echo "=================================================================="
echo "Searching for PhaseGPT Spiral Checkpoint"
echo "=================================================================="
echo ""

echo "[1/4] Searching for LoRA adapters (adapter_model.safetensors)..."
mdfind 'kMDItemFSName == "adapter_model.safetensors"c' 2>/dev/null | head -50 || echo "  (mdfind not available or no results)"

echo ""
echo "[2/4] Searching for merged models (model.safetensors > 100MB)..."
mdfind 'kMDItemFSName == "model.safetensors"c' 2>/dev/null | while read -r f; do
    size=$(stat -f%z "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 100000000 ]; then  # > 100MB
        echo "  $f ($(( size / 1024 / 1024 )) MB)"
    fi
done | head -50

echo ""
echo "[3/4] Searching for PhaseGPT/Spiral named files..."
mdfind 'kMDItemFSName == "*phasegpt*"c || kMDItemFSName == "*spiral*"c' 2>/dev/null | \
    grep -v ".venv\|node_modules\|__pycache__" | head -80

echo ""
echo "[4/4] Deep search in common locations..."
for d in "$HOME" \
         "$HOME/.cache/huggingface/hub" \
         "$HOME/Library/Application Support/LM Studio/models" \
         "/Volumes"; do
    if [ -d "$d" ]; then
        echo ""
        echo ">>> Searching $d"
        find "$d" -type f \( -name "adapter_model.safetensors" -o -name "*.gguf" \) \
            -size +5M 2>/dev/null | head -20 || true
    fi
done

echo ""
echo "=================================================================="
echo "Search complete!"
echo "=================================================================="
echo ""
echo "Next steps:"
echo "  1. Look for candidates in the output above"
echo "  2. For each LoRA candidate, check metadata:"
echo "     ls -lh /path/to/lora/adapter_model.safetensors"
echo "     cat /path/to/lora/adapter_config.json"
echo "     cat /path/to/lora/README.md"
echo ""
echo "  3. For GGUF files, check size and date:"
echo "     ls -lh /path/to/file.gguf"
echo "     shasum -a 256 /path/to/file.gguf"
echo ""
echo "  4. Once found, export with:"
echo "     ADAPTER=/path/to/spiral/lora OUT_NAME=phasegpt-v13-spiral make lmstudio-export"
echo ""
