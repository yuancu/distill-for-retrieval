# Distributed Training with DDP (Config-Based)

This guide explains how to use YAML configuration files for distributed training with DistributedDataParallel (DDP).

## Quick Start

### List Available Configs
```bash
python train_ddp.py --list-configs
```

### Single GPU Training
```bash
python train_ddp.py --config configs/default.yaml
```

### Multi-GPU Training (8 GPUs)
```bash
# Using shell script
./run_ddp.sh 8 configs/default.yaml

# OR using torchrun directly
torchrun --nproc_per_node=8 train_ddp.py --config configs/default.yaml
```

### Quick Test
```bash
# Fast test with minimal data (good for debugging)
torchrun --nproc_per_node=8 train_ddp.py --config configs/quick_test.yaml
```

## Configuration Files

All configuration files are stored in `configs/` directory:

- **`default.yaml`** - Default projection-based distillation (768d → 3584d)
- **`mrl.yaml`** - MRL-based distillation without projection (768d only)
- **`quick_test.yaml`** - Fast test with minimal data for debugging

### Config File Structure

```yaml
# Model configuration
model:
  teacher: "infly/inf-retriever-v1-pro"
  student: "sentence-transformers/all-mpnet-base-v2"
  teacher_dim: 3584
  student_dim: 768
  projection_hidden_dim: 1536
  use_projection: true  # true=projection, false=MRL

# Phase 1: General Distillation
phase1:
  batch_size: 64
  learning_rate: 2.0e-5
  warmup_steps: 1000
  num_epochs: 1
  mse_weight: 0.4
  cosine_weight: 0.6
  max_length: 512
  gradient_accumulation_steps: 4
  max_samples_per_dataset: 100000

# Phase 2: Task-Specific Training
phase2:
  batch_size: 64
  learning_rate: 5.0e-6
  warmup_steps: 500
  num_epochs: 3
  infonce_weight: 0.8
  mse_weight: 0.2
  temperature: 0.02
  max_length: 512
  num_negatives: 7
  gradient_accumulation_steps: 8
  max_samples: 500000

# Training control
training:
  skip_phase1: false
  skip_phase2: false
  save_to_artifacts: false

# Output paths
paths:
  output_dir: "./checkpoints"
  artifacts_dir: "./artifacts"
```

## Creating Custom Configs

1. **Copy an existing config:**
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. **Edit the config:**
   ```yaml
   # Example: Longer training with higher learning rate
   phase1:
     num_epochs: 2
     learning_rate: 3.0e-5

   phase2:
     num_epochs: 5
     learning_rate: 1.0e-5
   ```

3. **Run with your config:**
   ```bash
   torchrun --nproc_per_node=8 train_ddp.py --config configs/my_experiment.yaml
   ```

## Output Organization

Checkpoints and artifacts are automatically organized by experiment name:

```
checkpoints/
  distilled-mpnet-3584d-default/      # Experiment name = model-config
    config_default.yaml               # Config copy
    phase1_best.pt                    # Phase 1 checkpoint
    phase2_best.pt                    # Phase 2 checkpoint

artifacts/
  distilled-mpnet-3584d-default/      # Same experiment name
    config_default.yaml               # Config copy
    config_sentence_transformers.json
    model.safetensors
    ...
```

**Experiment name format:** `{model_name}-{config_name}`
- **model_name**: Auto-generated from config (e.g., `distilled-mpnet-3584d`)
- **config_name**: Config filename without `.yaml` (e.g., `default`)

## Command-Line Interface

```bash
python train_ddp.py [OPTIONS]

Options:
  --config PATH         Path to YAML config file [default: configs/default.yaml]
  --list-configs        List available config files and exit
  -h, --help           Show help message
```

## Examples

### Example 1: Default Training (8 GPUs)
```bash
torchrun --nproc_per_node=8 train_ddp.py
```
- Uses `configs/default.yaml`
- Projection-based (768d → 3584d)
- Saves to `checkpoints/distilled-mpnet-3584d-default/`

### Example 2: MRL Training (8 GPUs)
```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/mrl.yaml
```
- MRL-based distillation (no projection)
- Saves to `checkpoints/distilled-mpnet-768d-mrl-mrl/`

### Example 3: Quick Test (4 GPUs)
```bash
torchrun --nproc_per_node=4 train_ddp.py --config configs/quick_test.yaml
```
- Minimal data for fast testing
- Useful for debugging

### Example 4: Custom Config (8 GPUs)
```bash
# Create custom config
cat > configs/long_training.yaml << EOF
model:
  teacher: "infly/inf-retriever-v1-pro"
  student: "sentence-transformers/all-mpnet-base-v2"
  teacher_dim: 3584
  student_dim: 768
  projection_hidden_dim: 1536
  use_projection: true

phase1:
  batch_size: 64
  learning_rate: 2.0e-5
  num_epochs: 3  # More epochs
  # ... other settings

phase2:
  batch_size: 64
  learning_rate: 5.0e-6
  num_epochs: 5  # More epochs
  # ... other settings

training:
  skip_phase1: false
  skip_phase2: false
  save_to_artifacts: true  # Auto-save after training

paths:
  output_dir: "./checkpoints"
  artifacts_dir: "./artifacts"
EOF

# Run with custom config
torchrun --nproc_per_node=8 train_ddp.py --config configs/long_training.yaml
```

### Example 5: Resume from Phase 1
```yaml
# configs/phase2_only.yaml
training:
  skip_phase1: true   # Skip Phase 1
  skip_phase2: false  # Run Phase 2

# ... other settings
```

```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/phase2_only.yaml
```

## Performance

| Configuration | Time (Phase 1 + 2) | Speedup | GPU Utilization |
|--------------|------------------|---------|-----------------|
| Single GPU | ~32 hours | 1x | 100% (1 GPU) |
| 8 GPU DDP | ~4-5 hours | **7-7.5x** | ~95-98% (all) |
| 4 GPU DDP | ~8-9 hours | **3.8x** | ~95-98% (all) |

### Batch Size Tuning

**Effective batch size** = `batch_size` × `num_gpus`

Example with `batch_size: 64`:
- 1 GPU: effective batch size = 64
- 8 GPUs: effective batch size = 512

You may want to adjust learning rate accordingly:
- Single GPU: `learning_rate: 2e-5`
- 8 GPUs: Consider `learning_rate: 4e-5` (scaled)

## Monitoring

### Logs
Only rank 0 (GPU 0) prints detailed logs to avoid cluttering output.

### Progress Bars
Only rank 0 shows tqdm progress bars.

### Checkpoints
- Saved only by rank 0 to avoid race conditions
- Config file is automatically copied alongside checkpoints
- Best model is saved based on validation loss

## Troubleshooting

### Out of Memory
Reduce batch size in your config:
```yaml
phase1:
  batch_size: 32  # Reduced from 64
phase2:
  batch_size: 32  # Reduced from 64
```

### Config Not Found
```bash
# List available configs
python train_ddp.py --list-configs

# Check path
ls configs/
```

### NCCL Timeout
```bash
export NCCL_TIMEOUT=7200  # 2 hours
torchrun --nproc_per_node=8 train_ddp.py --config configs/default.yaml
```

### Port Already in Use
```bash
torchrun --nproc_per_node=8 --master_port=29501 train_ddp.py --config configs/default.yaml
```

## Multi-Node Training

For training across multiple machines:

```bash
# On machine 0 (master):
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=MASTER_IP \
    --master_port=29500 \
    train_ddp.py \
    --config configs/default.yaml

# On machine 1:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=MASTER_IP \
    --master_port=29500 \
    train_ddp.py \
    --config configs/default.yaml
```

## Best Practices

1. **Start with quick_test.yaml** for initial debugging
2. **Version control your configs** (commit to git)
3. **Use descriptive config names** (e.g., `high_lr_5epochs.yaml`)
4. **Document your experiments** in config comments
5. **Save configs with checkpoints** (done automatically)
6. **Test on 1-2 GPUs first**, then scale to 8 GPUs

## Migration from Old CLI-based Approach

**Old approach (deprecated):**
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --use_projection=True \
    --batch_size=64 \
    --phase1_epochs=1 \
    --phase2_epochs=3
```

**New config-based approach:**
```yaml
# configs/my_training.yaml
model:
  use_projection: true

phase1:
  batch_size: 64
  num_epochs: 1

phase2:
  batch_size: 64
  num_epochs: 3
```

```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/my_training.yaml
```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [YAML syntax reference](https://yaml.org/spec/1.2.2/)
