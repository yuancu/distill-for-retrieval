"""Distillation module for retrieval model distillation."""

from .models import ProjectionLayer, StudentModelWithProjection
from .losses import Phase1Loss, Phase2Loss
from .datasets import Phase1Dataset, Phase2Dataset, phase2_collate_fn
from .train import train_phase1, train_phase2
from .evaluate import evaluate_retrieval

__all__ = [
    'ProjectionLayer',
    'StudentModelWithProjection',
    'Phase1Loss',
    'Phase2Loss',
    'Phase1Dataset',
    'Phase2Dataset',
    'phase2_collate_fn',
    'train_phase1',
    'train_phase2',
    'evaluate_retrieval',
]
