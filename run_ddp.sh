#!/bin/bash
# Config-based distributed training launcher script

# Number of GPUs to use (default: all available)
NGPUS=${1:-8}

# Config file (default: configs/default.yaml)
CONFIG=${2:-configs/default.yaml}

echo "=========================================="
echo "Launching Distributed Training"
echo "=========================================="
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "=========================================="
echo

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found!"
    echo
    echo "Available configs:"
    python train_ddp.py --list-configs
    exit 1
fi

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    train_ddp.py \
    --config "$CONFIG"

echo
echo "=========================================="
echo "Training completed!"
echo "=========================================="
