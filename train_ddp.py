#!/usr/bin/env python
"""Distributed training launcher for multi-GPU training.

Usage:
    # Single GPU
    python train_ddp.py

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 train_ddp.py

    # Multi-GPU with custom config
    torchrun --nproc_per_node=8 train_ddp.py --use_projection=False
"""

import os
import argparse
import torch
import logging
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
    is_main_process,
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


logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed training for model distillation')

    # Model configuration
    parser.add_argument('--teacher_model', type=str, default='infly/inf-retriever-v1-pro',
                        help='Teacher model name')
    parser.add_argument('--student_model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Student model name')
    parser.add_argument('--use_projection', type=bool, default=True,
                        help='Use projection layer (True) or MRL-based distillation (False)')

    # Training configuration
    parser.add_argument('--phase1_epochs', type=int, default=1,
                        help='Number of epochs for Phase 1')
    parser.add_argument('--phase2_epochs', type=int, default=3,
                        help='Number of epochs for Phase 2')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU')
    parser.add_argument('--phase1_lr', type=float, default=2e-5,
                        help='Learning rate for Phase 1')
    parser.add_argument('--phase2_lr', type=float, default=5e-6,
                        help='Learning rate for Phase 2')

    # Output paths
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')

    # Dataset configuration
    parser.add_argument('--max_samples_phase1', type=int, default=100000,
                        help='Max samples per dataset for Phase 1')
    parser.add_argument('--max_samples_phase2', type=int, default=500000,
                        help='Max samples for Phase 2')

    # Skip phases
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Phase 1 training')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip Phase 2 training')

    # Model saving
    parser.add_argument('--save_to_artifacts', action='store_true',
                        help='Save final model to artifacts directory after training')
    parser.add_argument('--artifacts_dir', type=str, default='./artifacts',
                        help='Directory to save final model artifacts')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("Distributed Training Configuration")
        logger.info("=" * 80)
        logger.info(f"World size: {world_size}")
        logger.info(f"Teacher model: {args.teacher_model}")
        logger.info(f"Student model: {args.student_model}")
        logger.info(f"Distillation mode: {'WITH PROJECTION' if args.use_projection else 'MRL-BASED'}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
        logger.info("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    phase1_checkpoint = os.path.join(args.output_dir, "phase1_best")
    phase2_checkpoint = os.path.join(args.output_dir, "phase2_best")

    # Load teacher model (frozen, not wrapped with DDP)
    if is_main_process(rank):
        logger.info(f"Loading teacher model: {args.teacher_model}")

    teacher_model = SentenceTransformer(args.teacher_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(device)

    # Create student model
    if is_main_process(rank):
        logger.info(f"Loading student model: {args.student_model}")

    if args.use_projection:
        projection_layer = ProjectionLayer(768, 1536, 3584)
        student_model = StudentModelWithProjection(
            args.student_model,
            projection_layer=projection_layer,
            use_projection=True
        )
    else:
        student_model = StudentModelWithProjection(
            args.student_model,
            projection_layer=None,
            use_projection=False
        )

    # Wrap student model with DDP
    student_model = wrap_model_ddp(student_model, device, local_rank)

    if is_main_process(rank):
        logger.info("Models loaded and wrapped with DDP")

    # Phase 1 configuration
    phase1_config = {
        'batch_size': args.batch_size,
        'learning_rate': args.phase1_lr,
        'warmup_steps': 1000,
        'num_epochs': args.phase1_epochs,
        'mse_weight': 0.4,
        'cosine_weight': 0.6,
        'max_length': 512,
        'gradient_accumulation_steps': 4,
        'max_samples_per_dataset': args.max_samples_phase1
    }

    # Phase 2 configuration
    phase2_config = {
        'batch_size': args.batch_size,
        'learning_rate': args.phase2_lr,
        'warmup_steps': 500,
        'num_epochs': args.phase2_epochs,
        'infonce_weight': 0.8,
        'mse_weight': 0.2,
        'temperature': 0.02,
        'max_length': 512,
        'num_negatives': 7,
        'gradient_accumulation_steps': 8,
        'max_samples': args.max_samples_phase2
    }

    # Phase 1: General Distillation
    if not args.skip_phase1:
        if is_main_process(rank):
            logger.info("\nStarting Phase 1 training...")

        student_model = train_phase1(
            student_model,
            teacher_model,
            phase1_config,
            device,
            phase1_checkpoint,
            rank=rank
        )

        if is_main_process(rank):
            logger.info(f"Phase 1 complete. Checkpoint saved to: {phase1_checkpoint}.pt")
    else:
        if is_main_process(rank):
            logger.info("Skipping Phase 1 (--skip_phase1 flag set)")

    # Phase 2: Task-Specific Training
    if not args.skip_phase2:
        if is_main_process(rank):
            logger.info("\nStarting Phase 2 training...")

        student_model = train_phase2(
            student_model,
            teacher_model,
            phase2_config,
            device,
            phase2_checkpoint,
            rank=rank
        )

        if is_main_process(rank):
            logger.info(f"Phase 2 complete. Checkpoint saved to: {phase2_checkpoint}.pt")
    else:
        if is_main_process(rank):
            logger.info("Skipping Phase 2 (--skip_phase2 flag set)")

    # Save to artifacts (only on rank 0)
    if args.save_to_artifacts and is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Saving model to artifacts...")
        logger.info("=" * 80)

        # Determine which checkpoint to save (Phase 2 if available, else Phase 1)
        if not args.skip_phase2:
            final_checkpoint = f"{phase2_checkpoint}.pt"
            logger.info(f"Using Phase 2 checkpoint: {final_checkpoint}")
        elif not args.skip_phase1:
            final_checkpoint = f"{phase1_checkpoint}.pt"
            logger.info(f"Using Phase 1 checkpoint: {final_checkpoint}")
        else:
            logger.warning("No training was performed, skipping model saving")
            final_checkpoint = None

        if final_checkpoint and os.path.exists(final_checkpoint):
            try:
                output_path = save_distilled_model_to_artifacts(
                    student_model=student_model,  # Will be automatically unwrapped
                    checkpoint_path=final_checkpoint,
                    artifacts_dir=args.artifacts_dir,
                    model_name=None  # Auto-generate based on mode
                )
                logger.info(f"\n✓ Model saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        elif final_checkpoint:
            logger.warning(f"Checkpoint not found: {final_checkpoint}")

    # Cleanup
    cleanup_distributed()

    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
