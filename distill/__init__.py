"""Distillation module for retrieval model distillation."""

from .models import ProjectionLayer, StudentModelWithProjection
from .losses import Phase1Loss, Phase2Loss
from .datasets import Phase1Dataset, Phase2Dataset, phase2_collate_fn
from .train import train_phase1, train_phase2
from .evaluate import evaluate_retrieval
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    wrap_model_ddp,
    get_ddp_model,
    save_checkpoint,
)
from .config import TrainingConfig, load_config, list_available_configs

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
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'get_rank',
    'get_world_size',
    'wrap_model_ddp',
    'get_ddp_model',
    'save_checkpoint',
    'TrainingConfig',
    'load_config',
    'list_available_configs',
]
