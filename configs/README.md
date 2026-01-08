# Training Configuration Files

This directory contains YAML configuration files for distillation training. Each config file defines all hyperparameters, model settings, and training options.

## Available Configs

### `default.yaml` - Production Training
**Default projection-based distillation**
- Mode: WITH PROJECTION (768d → 3584d)
- Phase 1: 1 epoch, 100k samples
- Phase 2: 3 epochs, 500k samples
- Best for: Maximum quality retrieval models

### `mrl.yaml` - Efficient Training
**MRL-based distillation without projection**
- Mode: MRL (768d, no projection)
- Phase 1: 1 epoch, 100k samples
- Phase 2: 3 epochs, 500k samples
- Best for: Efficient deployment, lower computational cost

### `quick_test.yaml` - Debugging
**Fast test configuration**
- Minimal data: 1k samples per phase
- Single epoch for both phases
- Best for: Quick debugging, testing changes

## Creating Custom Configs

1. **Start from a template:**
   ```bash
   cp default.yaml my_experiment.yaml
   ```

2. **Edit your config:**
   ```yaml
   # my_experiment.yaml
   model:
     use_projection: true

   phase1:
     num_epochs: 2  # Increase epochs
     learning_rate: 3.0e-5  # Higher LR

   phase2:
     num_epochs: 5
     batch_size: 96  # Larger batch
   ```

3. **Run your experiment:**
   ```bash
   torchrun --nproc_per_node=8 train_ddp.py --config configs/my_experiment.yaml
   ```

## Config File Structure

```yaml
# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model:
  teacher: "infly/inf-retriever-v1-pro"          # Teacher model HF path
  student: "sentence-transformers/all-mpnet-base-v2"  # Student model HF path
  teacher_dim: 3584                               # Teacher output dimension
  student_dim: 768                                # Student output dimension
  projection_hidden_dim: 1536                     # Projection hidden size
  use_projection: true                            # true=projection, false=MRL

# ============================================================================
# PHASE 1: GENERAL DISTILLATION
# ============================================================================
# Objective: Learn to mimic teacher's general embedding space
# Loss: MSE + Cosine similarity
# Data: Configurable BEIR datasets (msmarco, nfcorpus, nq, hotpotqa, etc.)
# ============================================================================
phase1:
  # Dataset configuration - can specify multiple datasets
  datasets:
    - name: "msmarco"               # Dataset name
      max_samples: 100000           # Max samples from this dataset
    - name: "nfcorpus"              # Add more datasets as needed
      max_samples: 10000

  # Training hyperparameters
  batch_size: 64                    # Batch size per GPU
  learning_rate: 2.0e-5             # Learning rate
  warmup_steps: 1000                # Warmup steps
  num_epochs: 1                     # Number of epochs
  mse_weight: 0.4                   # MSE loss weight
  cosine_weight: 0.6                # Cosine loss weight
  max_length: 512                   # Max sequence length
  gradient_accumulation_steps: 4    # Gradient accumulation

# ============================================================================
# PHASE 2: TASK-SPECIFIC TRAINING
# ============================================================================
# Objective: Fine-tune for retrieval with contrastive learning
# Loss: InfoNCE + MSE
# Data: Configurable datasets with hard negatives (supports multiple datasets)
# ============================================================================
phase2:
  # Dataset configuration - can specify multiple datasets
  datasets:
    - name: "msmarco"               # Dataset name
      max_samples: 500000           # Max samples from this dataset
    - name: "nfcorpus"              # Add more datasets as needed
      max_samples: 50000

  # Training hyperparameters
  batch_size: 64                    # Batch size per GPU
  learning_rate: 5.0e-6             # Lower LR for fine-tuning
  warmup_steps: 500                 # Warmup steps
  num_epochs: 3                     # Number of epochs
  infonce_weight: 0.8               # InfoNCE loss weight
  mse_weight: 0.2                   # MSE loss weight
  temperature: 0.02                 # Temperature for InfoNCE
  max_length: 512                   # Max sequence length
  num_negatives: 7                  # Number of hard negatives
  gradient_accumulation_steps: 8    # Gradient accumulation

# ============================================================================
# TRAINING CONTROL
# ============================================================================
training:
  skip_phase1: false        # Skip Phase 1 if already trained
  skip_phase2: false        # Skip Phase 2 if not needed
  save_to_artifacts: false  # Auto-save to artifacts after training

# ============================================================================
# OUTPUT PATHS
# ============================================================================
# Experiment name: {model_name}-{config_name}
# Example: distilled-mpnet-3584d-default
# ============================================================================
paths:
  output_dir: "./checkpoints"   # Checkpoints saved here
  artifacts_dir: "./artifacts"  # Final models saved here
```

## Available Datasets

### Supported BEIR Datasets

The training system uses the **beir library** to load datasets from the BEIR benchmark. Datasets are automatically downloaded and cached locally in `./beir_datasets/` for efficient reuse.

**Requirements:**
```bash
pip install beir
```

**General Domain:**
- `msmarco` - MS MARCO (large-scale web search)
- `nq` - Natural Questions (Google search queries)
- `hotpotqa` - HotpotQA (multi-hop reasoning)
- `quora` - Quora duplicate questions

**Scientific/Technical:**
- `scidocs` - Scientific document retrieval
- `scifact` - Scientific fact verification
- `trec-covid` - COVID-19 research articles

**Specialized:**
- `nfcorpus` - Nutrition facts corpus
- `fiqa` - Financial question answering
- `arguana` - Argumentative QA
- `dbpedia-entity` - DBpedia entity retrieval
- `fever` - Fact extraction and verification
- `climate-fever` - Climate change fact verification
- `touche-2020` - Argument retrieval
- `signal1m` - Signal-1M (large-scale news retrieval)

**Dataset Loading:**
- All datasets are loaded using the official `beir` library
- Datasets are downloaded from the BEIR repository and cached locally
- Supports corpus (documents), queries, and qrels (relevance judgments)
- **Phase 1**: Randomly samples queries and documents separately with 1:19 ratio
  - Queries get instruction prefix: `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`
  - Documents have no prefix (just title + text)
  - No pairing required - purely for distillation
  - All texts treated uniformly during training (no query/document discrimination)
- **Phase 2**: Uses query-positive pairs with hard negatives from qrels
  - Proper relevance pairs for contrastive learning
  - Hard negatives extracted from qrels or randomly sampled

### Using Multiple Datasets in Phase 1

Phase 1 supports combining multiple datasets for better generalization:

```yaml
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 100000
    - name: "nfcorpus"
      max_samples: 10000
    - name: "hotpotqa"
      max_samples: 50000
```

**Benefits of multiple datasets:**
- Better generalization across domains
- Improved zero-shot performance
- More robust embeddings

**Tips:**
- Start with MS MARCO as the base (largest, most general)
- Add domain-specific datasets based on your use case
- Balance sample sizes to avoid dataset bias

### Using Multiple Datasets in Phase 2

Phase 2 also supports combining multiple datasets for task-specific fine-tuning:

```yaml
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 500000
    - name: "nfcorpus"
      max_samples: 50000
    - name: "hotpotqa"
      max_samples: 100000
  num_negatives: 7
```

**Benefits of multiple datasets in Phase 2:**
- Better generalization across domains and tasks
- More diverse hard negatives for contrastive learning
- Improved robustness to different query patterns

**Choosing Phase 2 datasets:**
- Use `msmarco` as base for general web search
- Add domain-specific datasets for your target application
- Balance sample sizes based on task importance
- All datasets will use contrastive learning with hard negatives

## Hyperparameter Tuning Guide

### Learning Rate
```yaml
# Conservative (safer, slower convergence)
phase1:
  learning_rate: 1.0e-5
phase2:
  learning_rate: 2.0e-6

# Aggressive (faster, may be unstable)
phase1:
  learning_rate: 5.0e-5
phase2:
  learning_rate: 1.0e-5
```

### Batch Size
```yaml
# Small (less memory, noisier gradients)
phase1:
  batch_size: 32
phase2:
  batch_size: 32

# Large (more memory, smoother gradients)
phase1:
  batch_size: 128
phase2:
  batch_size: 128
```

### Training Duration
```yaml
# Quick (1-2 hours on 8 GPUs)
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 10000
  num_epochs: 1
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 10000
  num_epochs: 1

# Standard (4-5 hours on 8 GPUs)
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 100000
  num_epochs: 1
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 500000
  num_epochs: 3

# Thorough (10-12 hours on 8 GPUs)
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 500000
  num_epochs: 3
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 1000000
  num_epochs: 5
```

## Common Use Cases

### 1. Quick Prototyping
```yaml
# File: configs/prototype.yaml
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 5000
  num_epochs: 1

phase2:
  datasets:
    - name: "msmarco"
      max_samples: 5000
  num_epochs: 1

training:
  save_to_artifacts: false
```

### 2. Production Training with Multiple Datasets
```yaml
# File: configs/production.yaml
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 500000
    - name: "nfcorpus"
      max_samples: 50000
    - name: "hotpotqa"
      max_samples: 100000
  num_epochs: 2
  learning_rate: 2.0e-5

phase2:
  datasets:
    - name: "msmarco"
      max_samples: 800000
    - name: "nfcorpus"
      max_samples: 100000
    - name: "hotpotqa"
      max_samples: 100000
  num_epochs: 5
  learning_rate: 5.0e-6

training:
  save_to_artifacts: true
```

### 3. Domain-Specific Training
```yaml
# File: configs/scientific.yaml
# Train on scientific/biomedical datasets
phase1:
  datasets:
    - name: "scidocs"
      max_samples: 50000
    - name: "scifact"
      max_samples: 10000
    - name: "trec-covid"
      max_samples: 20000
  num_epochs: 2

phase2:
  datasets:
    - name: "scifact"
      max_samples: 30000
    - name: "trec-covid"
      max_samples: 20000
  num_epochs: 3
```

### 4. Resume from Phase 1
```yaml
# File: configs/phase2_only.yaml
training:
  skip_phase1: true
  skip_phase2: false

# Use existing Phase 1 checkpoint
# Make sure checkpoint exists in output_dir
```

### 5. Ablation Study
```yaml
# File: configs/ablation_high_lr.yaml
phase1:
  learning_rate: 1.0e-4  # 5x higher than default

# File: configs/ablation_large_batch.yaml
phase1:
  batch_size: 256  # 4x larger than default

# File: configs/ablation_long_training.yaml
phase1:
  num_epochs: 5
phase2:
  num_epochs: 10
```

## Naming Conventions

Use descriptive names that indicate the key changes:
- `high_lr.yaml` - Higher learning rate
- `long_training.yaml` - More epochs
- `large_batch.yaml` - Larger batch size
- `mrl_fast.yaml` - MRL mode with fewer samples
- `ablation_*.yaml` - Ablation studies

## Validation

Test your config before full training:
```bash
# Dry run (will fail early if config is invalid)
python train_ddp.py --config configs/your_config.yaml

# Quick test (will train on minimal data)
python train_ddp.py --config configs/quick_test.yaml
```

## Version Control

**Always commit your configs to git:**
```bash
git add configs/my_experiment.yaml
git commit -m "Add config for high LR experiment"
```

This ensures reproducibility and tracks experiment history.

## Tips

1. **Start small**: Use `quick_test.yaml` to verify everything works
2. **Document changes**: Add comments explaining why you changed hyperparameters
3. **One config per experiment**: Don't reuse configs for different experiments
4. **Descriptive names**: Use names that describe the key difference
5. **Version control**: Always commit configs to git
6. **Copy, don't modify**: Copy existing configs instead of modifying them

## See Also

- `../README_DDP.md` - Complete DDP training guide
- `../train_ddp.py` - Training script documentation
- `../run_ddp.sh` - Launch script
