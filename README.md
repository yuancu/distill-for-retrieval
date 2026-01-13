# Distillation Framework for Retrieval Models

Knowledge distillation framework for creating fast, compact retrieval models from large teacher models with optional dynamic routing.

## Overview

This framework distills large embedding models (e.g., infly/inf-retriever-v1, 7B parameters) into smaller, faster student models (e.g., all-mpnet-base-v2, 110M parameters) while retaining 90-95% of the retrieval quality. The distilled models can be optionally combined with a trained router to create dynamic routing systems that intelligently switch between student and teacher models based on query difficulty.

### Key Features

- **Two-Phase Distillation**: General semantic alignment (Phase 1) + Task-specific training with hard negatives (Phase 2)
- **Pre-computed Embeddings**: Cache teacher embeddings to avoid repeated inference and speed up training
- **Multi-GPU Training**: Distributed training with DDP for faster convergence
- **Flexible Distillation Modes**:
  - **Projection-based**: Student (768d) → Projection layer → Teacher dimension (3584d)
  - **MRL-based**: Direct distillation to student's native dimension (768d)
- **Dynamic Routing** (Optional): Train a router to intelligently select between student/teacher at inference time
- **SentenceTransformer Compatible**: All outputs are compatible with the sentence-transformers library

## Quick Start

### 1. Pre-compute Teacher Embeddings (Recommended)

Pre-computing teacher embeddings significantly speeds up training and reduces GPU memory requirements.

```bash
torchrun --nproc_per_node=8 precompute_embeddings.py \
  --config configs/mpnet.yaml \
  --dataset msmarco \
  --batch-size 3072 \
  --max-length 128 \
  --full-dim
```

**Parameters:**
- `--config`: Training configuration file (determines teacher model, dimensions, etc.)
- `--dataset`: BEIR dataset name (e.g., msmarco, nfcorpus, scifact)
- `--batch-size`: Encoding batch size (adjust based on GPU memory)
- `--max-length`: Maximum sequence length for tokenization
- `--full-dim`: Store full-dimensional embeddings (use if training with projection)
- `--precision`: fp16 (default) or fp32

**Output:** Saves memory-mapped embeddings to `./cache/embedding/{dataset}_{precision}/`

### 2. Train Distilled Model with Router

```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/mpnet.yaml
```

This command:
- Loads configuration from `configs/mpnet.yaml`
- Runs two-phase distillation training
- Trains a router (if `train_router: true` in config)
- Saves checkpoints to `./checkpoints/`
- Saves final model to `./artifacts/`

**Available Configs:**
```bash
python train_ddp.py --list-configs
```

- `configs/mpnet.yaml` - MPNet student with MRL-based distillation + router training
- `configs/default.yaml` - Projection-based distillation (768d → 3584d)
- `configs/mrl.yaml` - MRL-based distillation (direct 768d)
- `configs/quick_test.yaml` - Quick test configuration with minimal data

### 3. Create Routed Model (Optional)

If you trained a router in Phase 2, create a SentenceTransformer-compatible routed model:

```bash
python create_routed_model.py \
  artifacts/distilled-768d-mrl-mpnet \
  infly/inf-retriever-v1-pro \
  0.5 \
  artifacts/routed-model
```

**Parameters:**
- First arg: Path to student model (must contain `router.pt`)
- Second arg: Teacher model name or path
- Third arg: Routing threshold (0-1, higher = less teacher usage)
- Fourth arg: Output directory

### 4. Evaluate Model

Evaluation is performed in a separate repository:

```bash
cd /path/to/inf_x_retriever
python run_beir.py --dataset nfcorpus --model /path/to/artifacts/distilled-model
```

Evaluation repo: [https://github.com/yuancu/inf_x_retriever](https://github.com/yuancu/inf_x_retriever)

## Project Structure

```
distill/
├── configs/                    # Training configurations
│   ├── mpnet.yaml             # MPNet + MRL + Router (recommended)
│   ├── default.yaml           # Projection-based distillation
│   ├── mrl.yaml               # MRL-based distillation
│   └── quick_test.yaml        # Quick test configuration
├── distill/                    # Core package
│   ├── models.py              # StudentModelWithProjection, Router, RoutingModule
│   ├── train.py               # Phase 1/2 training loops
│   ├── losses.py              # MSE, Cosine, InfoNCE losses
│   ├── datasets.py            # BEIR dataset loaders
│   ├── config.py              # Config management
│   ├── distributed.py         # DDP utilities
│   └── save_model.py          # Model saving utilities
├── precompute_embeddings.py   # Pre-compute teacher embeddings
├── train_ddp.py               # Main training script
├── create_routed_model.py     # Create routed model
└── artifacts/                 # Saved models (output)
```

## Training Configuration

### Config File Structure

```yaml
# Query instruction prefix (for retrieval tasks)
query_instruction: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

# Pre-computed embeddings directory
precomputed_embeddings_dir: "./cache/embedding/"

# Model configuration
model:
  teacher: "infly/inf-retriever-v1"
  student: "sentence-transformers/all-mpnet-base-v2"
  teacher_dim: 3584
  student_dim: 768
  projection_hidden_dim: 1536  # Only used if use_projection=true
  use_projection: false        # false = MRL-based, true = projection-based

# Phase 1: General Distillation
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 10000000
  batch_size: 64
  learning_rate: 2.0e-5
  warmup_steps: 1000
  num_epochs: 3
  mse_weight: 1.0
  cosine_weight: 0.0
  max_length: 512
  gradient_accumulation_steps: 4

# Phase 2: Task-Specific Training with Router
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 10000000
  batch_size: 16
  learning_rate: 5.0e-6
  warmup_steps: 500
  num_epochs: 3
  infonce_weight: 0.8
  mse_weight: 0.2
  temperature: 0.02
  max_length: 512
  num_negatives: 7
  gradient_accumulation_steps: 8

  # Router configuration
  train_router: true           # Enable router training
  router_loss_weight: 0.1      # Router loss weight
  router_warmup_ratio: 0.3     # Start router training after 30% of steps
  router_lr: 1.0e-4           # Router learning rate

# Training control
training:
  skip_phase1: false           # Set to true to skip Phase 1
  skip_phase2: false           # Set to true to skip Phase 2
  save_to_artifacts: true      # Save final model to artifacts/

# Output paths
paths:
  output_dir: "./checkpoints"
  artifacts_dir: "./artifacts"
```

### Distillation Modes

#### MRL-Based (Recommended)
```yaml
model:
  use_projection: false
```
- Distills directly to student's native dimension (768d)
- Teacher embeddings are truncated to 768d
- Faster training and inference
- Smaller model size (~110M parameters)

#### Projection-Based
```yaml
model:
  use_projection: true
  projection_hidden_dim: 1536
```
- Projects student embeddings to teacher dimension (3584d)
- Compatible with teacher embeddings
- Enables hybrid systems
- Larger model size (~117M parameters)

## Training Phases

### Phase 1: General Semantic Alignment

Trains the student model to match teacher embeddings on large-scale data using MSE and/or Cosine loss.

**Goal:** Learn general semantic representations

**Loss Functions:**
- MSE Loss: `||student_emb - teacher_emb||²`
- Cosine Loss: `1 - cosine_similarity(student_emb, teacher_emb)`

### Phase 2: Task-Specific Training with Hard Negatives

Trains on retrieval tasks with InfoNCE loss and hard negatives, optionally training a router.

**Goal:** Learn retrieval-specific features and query difficulty estimation

**Loss Functions:**
- InfoNCE Loss: Contrastive learning with hard negatives
- MSE Loss: Continued alignment with teacher
- Router Loss (optional): Predicts student-teacher MSE + InfoNCE difference

**Router Training:**
The router is a small MLP (768d → 256 → 128 → 1) that predicts query difficulty:
- Input: Student query embeddings
- Output: Difficulty score [0, 1]
- Training signal: MSE + InfoNCE difference between student and teacher

## Using Trained Models

### Loading Distilled Model

```python
from sentence_transformers import SentenceTransformer

# Load student model (768d embeddings)
model = SentenceTransformer("artifacts/distilled-768d-mrl-mpnet")
embeddings = model.encode(["your text here"])
print(embeddings.shape)  # (1, 768)
```

### Loading Routed Model

```python
from sentence_transformers import SentenceTransformer

# Load routed model (automatically switches between student/teacher)
model = SentenceTransformer("artifacts/routed-model")

# Encode - routing happens automatically
embeddings = model.encode([
    "What is machine learning?",
    "The weather is nice today."
])

# Check routing decisions (Index [2] is the RoutingModule)
routing = model[2].last_routing_decisions
print(f"Teacher usage: {routing.float().mean()*100:.1f}%")
# Output: Teacher usage: 45.0%

# Adjust threshold dynamically
model[2].threshold = 0.7  # Higher = less teacher usage
embeddings = model.encode(["Query"])
```

### Comparing with Teacher

```python
from sentence_transformers import SentenceTransformer
import torch

# Load both models
teacher = SentenceTransformer("infly/inf-retriever-v1-pro")
student = SentenceTransformer("artifacts/distilled-768d-mrl-mpnet")

texts = ["sample text 1", "sample text 2"]

# Encode with both
teacher_embs = teacher.encode(texts, convert_to_tensor=True)
student_embs = student.encode(texts, convert_to_tensor=True)

# Compare (using first 768 dimensions of teacher)
teacher_embs_768d = teacher_embs[:, :768]
similarity = torch.cosine_similarity(teacher_embs_768d, student_embs, dim=-1)
print(f"Average alignment: {similarity.mean().item():.4f}")
```

## Advanced Usage

### Custom Dataset

Add your dataset to Phase 1/2 configs:

```yaml
phase1:
  datasets:
    - name: "msmarco"
      max_samples: 5000000
    - name: "nfcorpus"
      max_samples: 100000
```

### Resume Training

Training automatically saves checkpoints. To resume:

```bash
# Modify config to skip completed phases
training:
  skip_phase1: true  # Skip if Phase 1 is complete
  skip_phase2: false
```

### Router Threshold Tuning

After training, test different thresholds to find the optimal speed/quality trade-off:

```python
import numpy as np

thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    model[2].threshold = threshold
    embeddings = model.encode(test_queries)
    routing = model[2].last_routing_decisions
    teacher_pct = routing.float().mean().item() * 100
    print(f"Threshold: {threshold}, Teacher usage: {teacher_pct:.1f}%")
```

### Multi-Dataset Training

Train on multiple datasets simultaneously:

```yaml
phase2:
  datasets:
    - name: "msmarco"
      max_samples: 5000000
    - name: "hotpotqa"
      max_samples: 1000000
    - name: "nfcorpus"
      max_samples: 50000
```

## Performance Characteristics

| Model | Params | Embedding Dim | Relative Speed | Use Case |
|-------|--------|---------------|----------------|----------|
| Teacher (inf-retriever-v1-pro) | ~7B | 3584 | 1x | Best quality |
| Student (distilled) | ~110M | 768 | ~10x | Fast retrieval |
| Routed (threshold=0.5) | ~7.1B | 768 | ~5x | Balanced |

**Typical Results:**
- Student achieves 90-95% of teacher's NDCG@10 on BEIR benchmarks
- 70x fewer parameters, 10x faster inference
- Routed model adapts speed/quality dynamically based on query difficulty

## Troubleshooting

### Out of Memory During Training

1. Reduce batch size: `batch_size: 32`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Use pre-computed embeddings (avoids loading teacher model)
4. Use fp16 precision for pre-computed embeddings

### Pre-computed Embeddings Not Found

Ensure the directory structure matches:
```
cache/embedding/
  msmarco_fp16/
    ├── queries.mmap
    ├── corpus.mmap
    └── metadata.json
```

Check config paths:
```yaml
precomputed_embeddings_dir: "./cache/embedding/"
```

### Router Training Fails

Ensure Phase 2 config has:
```yaml
phase2:
  train_router: true
  router_loss_weight: 0.1
  router_warmup_ratio: 0.3
  router_lr: 1.0e-4
```

Router requires both student and teacher models to compute difficulty targets.

### Model Loading Issues

```python
# Check saved model contents
import os
print(os.listdir("artifacts/distilled-768d-mrl-mpnet"))
# Should contain: modules.json, config.json, 0_Transformer/, 1_Pooling/, etc.
```

## License

[MIT]

