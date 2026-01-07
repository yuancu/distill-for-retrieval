# Distill Module

This module contains all the components for retrieval model distillation.

## Structure

```
distill/
├── __init__.py          # Module exports
├── models.py            # ProjectionLayer, StudentModelWithProjection
├── losses.py            # Phase1Loss, Phase2Loss
├── datasets.py          # Phase1Dataset, Phase2Dataset, phase2_collate_fn
├── train.py             # train_phase1, train_phase2
├── evaluate.py          # evaluate_retrieval
└── README.md            # This file
```

## Usage

### From Python script:
```python
from distill import (
    ProjectionLayer,
    StudentModelWithProjection,
    train_phase1,
    train_phase2,
    evaluate_retrieval
)
```

### From Jupyter Notebook:
See `distill_mpnet_driver.ipynb` for a complete example.

## Components

### Models (`models.py`)
- **ProjectionLayer**: 2-layer MLP (768→1536→3584) with LayerNorm and GELU
- **StudentModelWithProjection**: Wraps student model with projection layer

### Losses (`losses.py`)
- **Phase1Loss**: MSE + Cosine similarity loss for general distillation
- **Phase2Loss**: InfoNCE + MSE loss for task-specific training

### Datasets (`datasets.py`)
- **Phase1Dataset**: Loads MS MARCO (+ optionally NQ, HotpotQA)
- **Phase2Dataset**: Loads MS MARCO with hard negatives
- **phase2_collate_fn**: Custom collate function for variable-length negatives

### Training (`train.py`)
- **train_phase1**: General distillation training loop
- **train_phase2**: Task-specific training loop with contrastive learning

### Evaluation (`evaluate.py`)
- **evaluate_retrieval**: Evaluates student vs teacher on MS MARCO dev set

## Training Pipeline

1. **Phase 1**: General semantic matching
   - Loss: MSE (0.4) + Cosine (0.6)
   - Data: Diverse QA datasets
   - Goal: Learn general embedding alignment

2. **Phase 2**: Task-specific retrieval
   - Loss: InfoNCE (0.8) + MSE (0.2)
   - Data: MS MARCO with hard negatives
   - Goal: Optimize for ranking performance

## Output

- `student_base/`: Sentence-transformer model (768d embeddings)
- `projection_layer.pt`: Projection weights (768→3584)

Use projected 3584d embeddings for hybrid system that can switch between student and teacher models.
