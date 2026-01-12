#!/usr/bin/env python
"""Config-based distributed training launcher for multi-GPU training.

Usage:
    # Single GPU with default config
    python train_ddp.py

    # Multi-GPU (8 GPUs) with default config
    torchrun --nproc_per_node=8 train_ddp.py

    # Multi-GPU with custom config
    torchrun --nproc_per_node=8 train_ddp.py --config configs/mrl.yaml

    # List available configs
    python train_ddp.py --list-configs
"""

import os
import argparse
import torch
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

from distill import (
    ProjectionLayer,
    StudentModelWithProjection,
    train_phase1,
    train_phase2,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    TrainingConfig,
    load_config,
    list_available_configs,
)
from distill.save_model import save_distilled_model_to_artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][Rank %(rank)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class RankFilter(logging.Filter):
    """Add rank information to log records."""
    def filter(self, record):
        record.rank = get_rank()
        return True


# Add the rank filter to all handlers of the root logger
# This ensures all loggers (including third-party ones) have rank information
rank_filter = RankFilter()
for handler in logging.root.handlers:
    handler.addFilter(rank_filter)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Config-based distributed training for model distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU with default config
  python train_ddp.py

  # 8 GPUs with custom config
  torchrun --nproc_per_node=8 train_ddp.py --config configs/mrl.yaml

  # List available configs
  python train_ddp.py --list-configs

Config file structure:
  configs/
    default.yaml  - Default projection-based distillation
    mrl.yaml      - MRL-based distillation (no projection)
    quick_test.yaml - Quick test with minimal data
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML config file (default: configs/default.yaml)'
    )

    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available config files and exit'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # List configs if requested
    if args.list_configs:
        print("Available configuration files:")
        print("=" * 80)
        configs = list_available_configs()
        if not configs:
            print("No config files found in configs/")
        else:
            for config_path in configs:
                print(f"  - {config_path}")
                # Try to load and show brief description
                try:
                    cfg = TrainingConfig(str(config_path))
                    mode = "WITH PROJECTION" if cfg.model['use_projection'] else "MRL-BASED"
                    print(f"      Mode: {mode}")
                    print(f"      Phase 1 epochs: {cfg.phase1['num_epochs']}")
                    print(f"      Phase 2 epochs: {cfg.phase2['num_epochs']}")
                    print()
                except Exception as e:
                    print(f"      Error loading: {e}\n")
        return

    # Load configuration
    try:
        config = TrainingConfig(args.config)
    except Exception as e:
        logger.error(f"Failed to load config from {args.config}: {e}")
        logger.info("Use --list-configs to see available configs")
        return

    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("Distributed Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Config file: {args.config}")
        logger.info(f"Config name: {config.config_name}")
        logger.info(f"Experiment name: {config.get_experiment_name()}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Teacher model: {config.model['teacher']}")
        logger.info(f"Student model: {config.model['student']}")
        mode_str = "WITH PROJECTION" if config.model['use_projection'] else "MRL-BASED"
        logger.info(f"Distillation mode: {mode_str}")
        logger.info("=" * 80)

    # Create output directories and save config
    checkpoint_dir = config.get_checkpoint_dir()
    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.save_config_copy(checkpoint_dir)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    phase1_checkpoint = checkpoint_dir / "phase1_best"
    phase2_checkpoint = checkpoint_dir / "phase2_best"

    # Check if using pre-computed embeddings
    # The config method will generate the proper path based on dataset name
    precomputed_embeddings_dir = config.get_precomputed_embeddings_dir()

    # Load teacher model only if not using pre-computed embeddings
    teacher_model = None
    if precomputed_embeddings_dir is None:
        if is_main_process(rank):
            logger.info(f"\nLoading teacher model: {config.model['teacher']}")

        teacher_model = SentenceTransformer(config.model['teacher'])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model = teacher_model.to(device)
    else:
        if is_main_process(rank):
            logger.info(f"\nSkipping teacher model loading (using pre-computed embeddings from {precomputed_embeddings_dir})")

    # Create student model
    if is_main_process(rank):
        logger.info(f"Loading student model: {config.model['student']}")

    if config.model['use_projection']:
        projection_layer = ProjectionLayer(
            config.model['student_dim'],
            config.model['projection_hidden_dim'],
            config.model['teacher_dim']
        )
        student_model = StudentModelWithProjection(
            config.model['student'],
            projection_layer=projection_layer,
            use_projection=True
        )
    else:
        student_model = StudentModelWithProjection(
            config.model['student'],
            projection_layer=None,
            use_projection=False
        )

    # Wrap student model with DDP
    student_model = wrap_model_ddp(student_model, device, local_rank)

    if is_main_process(rank):
        logger.info("Models loaded and wrapped with DDP")

    # Phase 1: General Distillation
    if not config.training['skip_phase1']:
        if is_main_process(rank):
            logger.info("\n" + "=" * 80)
            logger.info("Starting Phase 1 training...")
            logger.info("=" * 80)

        student_model = train_phase1(
            student_model,
            teacher_model,
            config.phase1,
            device,
            str(phase1_checkpoint),
            rank=rank
        )

        if is_main_process(rank):
            logger.info(f"Phase 1 complete. Checkpoint saved to: {phase1_checkpoint}.pt")
    else:
        if is_main_process(rank):
            logger.info("Skipping Phase 1 (skip_phase1=true in config)")

    # Phase 2: Task-Specific Training
    if not config.training['skip_phase2']:
        if is_main_process(rank):
            logger.info("\n" + "=" * 80)
            logger.info("Starting Phase 2 training...")
            logger.info("=" * 80)

        result = train_phase2(
            student_model,
            teacher_model,
            config.phase2,
            device,
            str(phase2_checkpoint),
            rank=rank
        )

        # Unpack result - train_phase2 returns (model, router) if router training is enabled
        if isinstance(result, tuple):
            student_model, router = result
            if is_main_process(rank):
                logger.info("Router training was enabled and completed")
        else:
            student_model = result

        if is_main_process(rank):
            logger.info(f"Phase 2 complete. Checkpoint saved to: {phase2_checkpoint}.pt")
    else:
        if is_main_process(rank):
            logger.info("Skipping Phase 2 (skip_phase2=true in config)")

    # Synchronize all ranks before saving (prevents timeout during artifact saving)
    if get_world_size() > 1:
        torch.distributed.barrier()

    # Save to artifacts (only on rank 0)
    if config.training['save_to_artifacts'] and is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Saving model to artifacts...")
        logger.info("=" * 80)

        # Determine which checkpoint to save (Phase 2 if available, else Phase 1)
        if not config.training['skip_phase2']:
            final_checkpoint = f"{phase2_checkpoint}.pt"
            logger.info(f"Using Phase 2 checkpoint: {final_checkpoint}")
        elif not config.training['skip_phase1']:
            final_checkpoint = f"{phase1_checkpoint}.pt"
            logger.info(f"Using Phase 1 checkpoint: {final_checkpoint}")
        else:
            logger.warning("No training was performed, skip saving artifacts")
            final_checkpoint = None

        if final_checkpoint and os.path.exists(final_checkpoint):
            # Get artifacts directory
            artifacts_dir = config.get_artifacts_dir()
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save config copy to artifacts
            config.save_config_copy(artifacts_dir)

            try:
                # Model name will be auto-generated based on config
                model_name = config.get_experiment_name()

                output_path = save_distilled_model_to_artifacts(
                    student_model=student_model,
                    checkpoint_path=final_checkpoint,
                    artifacts_dir=str(artifacts_dir.parent),  # Parent dir, function creates subdirectory
                    model_name=model_name
                )
                logger.info(f"\n✓ Model saved to: {output_path}")
                logger.info(f"✓ Config saved to: {artifacts_dir / f'config_{config.config_name}.yaml'}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        elif final_checkpoint:
            logger.warning(f"Checkpoint not found: {final_checkpoint}")

    # Synchronize all ranks after saving (ensures rank 0 finishes before cleanup)
    if get_world_size() > 1:
        torch.distributed.barrier()

    # Cleanup
    cleanup_distributed()

    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.get_experiment_name()}")
        logger.info(f"Checkpoints: {checkpoint_dir}")
        if config.training['save_to_artifacts']:
            logger.info(f"Artifacts: {config.get_artifacts_dir()}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
