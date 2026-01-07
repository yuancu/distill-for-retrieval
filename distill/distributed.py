"""Distributed training utilities for multi-GPU training with DDP."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training environment.

    This function should be called at the start of each worker process.
    It reads environment variables set by torchrun/torch.distributed.launch.

    Returns:
        tuple: (rank, local_rank, world_size, device)
            - rank: Global rank of this process
            - local_rank: Local rank on this node
            - world_size: Total number of processes
            - device: torch.device for this process
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        # Not running in distributed mode
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # NCCL is best for GPU training
            init_method='env://',  # Use environment variables
        )

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        if rank == 0:
            print(f"Initialized DDP with {world_size} processes")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if rank == 0:
            print("Running in single-GPU mode")

    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


def get_rank():
    """Get the rank of current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def save_checkpoint(state_dict, path, rank):
    """Save checkpoint only on rank 0.

    Args:
        state_dict: State dictionary to save
        path: Path to save checkpoint
        rank: Current process rank
    """
    if is_main_process(rank):
        torch.save(state_dict, path)
        barrier()
    else:
        barrier()


def wrap_model_ddp(model, device, local_rank):
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device: Device to move model to
        local_rank: Local rank for DDP

    Returns:
        Wrapped model (DDP if distributed, otherwise original)
    """
    model = model.to(device)

    if get_world_size() > 1:
        # Wrap with DDP
        # find_unused_parameters=True is needed for models with conditional paths
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Set to True if you have unused parameters
        )

    return model


def get_ddp_model(model):
    """Get the underlying model from DDP wrapper.

    Args:
        model: Model (possibly wrapped with DDP)

    Returns:
        Underlying model without DDP wrapper
    """
    if isinstance(model, DDP):
        return model.module
    return model
