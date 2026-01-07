#!/bin/bash
# Distributed training launcher script

# Number of GPUs to use (default: all available)
NGPUS=${1:-8}

# Training mode (default: with projection)
USE_PROJECTION=${2:-True}

echo "=========================================="
echo "Launching Distributed Training"
echo "=========================================="
echo "Number of GPUs: $NGPUS"
echo "Projection mode: $USE_PROJECTION"
echo "=========================================="
echo

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    train_ddp.py \
    --use_projection=$USE_PROJECTION \
    --batch_size=64 \
    --phase1_epochs=1 \
    --phase2_epochs=3

echo
echo "=========================================="
echo "Training completed!"
echo "=========================================="
