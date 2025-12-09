#!/bin/bash
# PhaseGPT 7B Training Script (MLX)
# Target: mlx-community/Qwen2.5-7B-Instruct-4bit

MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
DATA_DIR="data"
ADAPTER_DIR="adapters/phasegpt_oracle_7b"

echo "Starting Training on $MODEL..."

# Flags optimized for 36GB Mac Studio:
# --batch-size 4: Safe for 7B (vs 1 for 14B)
# --lora-layers 16: Good balance of capacity/speed
# --iters 1000: Approx 1 epoch for 9k samples with batch 4 (9000/4 = 2250 steps, so 1000 is ~0.5 epoch check)

mlx_lm.lora \
    --model $MODEL \
    --train \
    --data $DATA_DIR \
    --batch-size 4 \
    --grad-checkpoint \
    --lora-layers 16 \
    --iters 600 \
    --learning-rate 1e-5 \
    --adapter-path $ADAPTER_DIR \
    --save-every 100 \
    --seed 42
