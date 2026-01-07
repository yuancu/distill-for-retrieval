# Distributed Training with DDP

This guide explains how to use DistributedDataParallel (DDP) for multi-GPU training.

## Quick Start

### Single GPU Training (default)
```bash
python train_ddp.py
```

### Multi-GPU Training (8 GPUs)
```bash
# Using shell script
./run_ddp.sh 8 True

# OR using torchrun directly
torchrun --nproc_per_node=8 train_ddp.py
```

### Multi-GPU with MRL mode (no projection)
```bash
./run_ddp.sh 8 False

# OR
torchrun --nproc_per_node=8 train_ddp.py --use_projection=False
```

## Command Line Arguments

```bash
python train_ddp.py [OPTIONS]

Options:
  --teacher_model TEXT          Teacher model name [default: infly/inf-retriever-v1-pro]
  --student_model TEXT          Student model name [default: sentence-transformers/all-mpnet-base-v2]
  --use_projection BOOL         Use projection layer [default: True]
  --batch_size INT              Batch size per GPU [default: 64]
  --phase1_epochs INT           Phase 1 epochs [default: 1]
  --phase2_epochs INT           Phase 2 epochs [default: 3]
  --phase1_lr FLOAT             Phase 1 learning rate [default: 2e-5]
  --phase2_lr FLOAT             Phase 2 learning rate [default: 5e-6]
  --output_dir TEXT             Output directory [default: ./checkpoints]
  --max_samples_phase1 INT      Max samples for Phase 1 [default: 100000]
  --max_samples_phase2 INT      Max samples for Phase 2 [default: 500000]
  --skip_phase1                 Skip Phase 1 training
  --skip_phase2                 Skip Phase 2 training
  --save_to_artifacts           Save final model to artifacts after training
  --artifacts_dir TEXT          Artifacts directory [default: ./artifacts]
```

## Examples

### Train with 4 GPUs, custom batch size
```bash
torchrun --nproc_per_node=4 train_ddp.py --batch_size=128
```

### Train only Phase 2 with 8 GPUs
```bash
torchrun --nproc_per_node=8 train_ddp.py --skip_phase1
```

### MRL mode with custom learning rate
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --use_projection=False \
    --phase1_lr=3e-5 \
    --phase2_lr=1e-5
```

### Train and save final model to artifacts
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --save_to_artifacts \
    --artifacts_dir=./my_models
```

This will:
1. Train the model with 8 GPUs
2. After training completes, save the final model to `./my_models/`
3. The saved model can be loaded with SentenceTransformer

## Performance

### Expected Speedup
- **8 GPUs**: ~7-7.5x speedup vs single GPU
- **4 GPUs**: ~3.8x speedup vs single GPU
- **2 GPUs**: ~1.9x speedup vs single GPU

### Effective Batch Size
With `--batch_size=64` and 8 GPUs, the effective batch size is **512** (64 × 8).

You may want to adjust the learning rate accordingly:
- Single GPU: batch_size=64, lr=2e-5
- 8 GPUs: batch_size=64 per GPU (effective 512), lr=4e-5 (scaled)

## Monitoring

### Logs
Only rank 0 (GPU 0) will print detailed logs. Other processes run silently.

### Progress Bars
Only rank 0 shows the tqdm progress bar to avoid cluttering the terminal.

### Checkpoints
Only rank 0 saves checkpoints to avoid race conditions.

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
torchrun --nproc_per_node=8 train_ddp.py --batch_size=32
```

### NCCL Timeout
Increase timeout (useful for slow networks):
```bash
export NCCL_TIMEOUT=7200  # 2 hours
torchrun --nproc_per_node=8 train_ddp.py
```

### Port Already in Use
Change master port:
```bash
torchrun --nproc_per_node=8 --master_port=29501 train_ddp.py
```

### Multi-Node Training
For training across multiple machines:
```bash
# On machine 0 (master):
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=MASTER_IP \
    --master_port=29500 \
    train_ddp.py

# On machine 1:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=MASTER_IP \
    --master_port=29500 \
    train_ddp.py
```

## Architecture

### How DDP Works
1. Each GPU runs its own process
2. Each process loads a copy of the model
3. Data is split across GPUs using DistributedSampler
4. Gradients are synchronized via All-Reduce after backward pass
5. All model copies stay synchronized

### Key Differences from Single GPU
- **Sampler**: Uses DistributedSampler instead of random shuffling
- **Model**: Wrapped with DDP for gradient synchronization
- **Logging**: Only rank 0 logs to avoid duplicate messages
- **Checkpointing**: Only rank 0 saves to avoid race conditions

## Comparison: Single GPU vs DDP

| Feature | Single GPU | 8 GPU DDP |
|---------|-----------|-----------|
| Training time (Phase 1) | ~8 hours | ~1-1.2 hours |
| Training time (Phase 2) | ~24 hours | ~3-3.5 hours |
| Batch size | 64 | 512 (64×8) |
| GPU utilization | 100% (1 GPU) | ~95-98% (8 GPUs) |
| Code changes | None | Minimal |

## Best Practices

1. **Batch Size**: Start with same per-GPU batch size as single GPU
2. **Learning Rate**: Scale linearly with number of GPUs (e.g., 2x GPUs → 2x LR)
3. **Gradient Accumulation**: May need to adjust for large effective batch sizes
4. **Debugging**: Test on single GPU first, then scale to multi-GPU
5. **Monitoring**: Use tensorboard or wandb for distributed training metrics

## Integration with Notebook

To use DDP from Jupyter notebook, save your configuration and launch from terminal:
```python
# In notebook: generate config
config = {
    'use_projection': True,
    'batch_size': 64,
    'phase1_epochs': 1,
    'phase2_epochs': 3,
}

import json
with open('ddp_config.json', 'w') as f:
    json.dump(config, f)
```

```bash
# In terminal: launch DDP training
torchrun --nproc_per_node=8 train_ddp.py
```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
