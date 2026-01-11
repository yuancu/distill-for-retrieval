"""Training functions for distillation."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import logging

from .datasets import Phase1DatasetPrecomputed, Phase2DatasetPrecomputed
from .losses import Phase1Loss, Phase2Loss
from .distributed import (
    is_main_process,
    get_world_size,
    save_checkpoint as save_checkpoint_dist,
    get_ddp_model,
)

logger = logging.getLogger(__name__)


class MultiDatasetWrapper:
    """Wrapper for multiple Phase1/Phase2 datasets that supports get_embedding methods."""

    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = [0]
        for ds in datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break

        # Get local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

    def get_embedding(self, idx):
        """Get embedding from the appropriate dataset."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break

        # Get local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx].get_embedding(local_idx)

    def get_query_embedding(self, idx):
        """Get query embedding (for Phase 2)."""
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx].get_query_embedding(local_idx)

    def get_positive_embedding(self, idx):
        """Get positive embedding (for Phase 2)."""
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx].get_positive_embedding(local_idx)

    def get_negative_embeddings(self, idx):
        """Get negative embeddings (for Phase 2)."""
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx].get_negative_embeddings(local_idx)


def phase2_collate_fn(batch):
    """Custom collate function to handle variable-length negatives"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives_list = [item['negatives'] for item in batch]

    # Get embedding indices if present
    query_indices = [item.get('_query_idx') for item in batch if '_query_idx' in item]
    pos_indices = [item.get('_pos_idx') for item in batch if '_pos_idx' in item]
    neg_indices_list = [item.get('_neg_indices', []) for item in batch]

    # Find max number of negatives in this batch
    max_negatives = max(len(negs) for negs in negatives_list)

    # Pad negatives to same length (repeat last negative if needed)
    padded_negatives = []
    padded_neg_indices = []
    for i, negs in enumerate(negatives_list):
        if len(negs) < max_negatives:
            # Pad by repeating the last negative
            padded = negs + [negs[-1]] * (max_negatives - len(negs))
            # Pad indices too
            neg_idx = neg_indices_list[i]
            if neg_idx:
                padded_idx = neg_idx + [neg_idx[-1]] * (max_negatives - len(neg_idx))
            else:
                padded_idx = []
        else:
            padded = negs
            padded_idx = neg_indices_list[i] if i < len(neg_indices_list) else []
        padded_negatives.append(padded)
        padded_neg_indices.append(padded_idx)

    result = {
        'query': queries,
        'positive': positives,
        'negatives': padded_negatives
    }

    # Add embedding indices if present
    if query_indices and all(idx is not None for idx in query_indices):
        result['_query_indices'] = query_indices
        result['_pos_indices'] = pos_indices
        result['_neg_indices'] = padded_neg_indices

    return result


def train_phase1(student_model, teacher_model, config, device, checkpoint_path, rank=0):
    """Phase 1: General distillation with MSE + Cosine loss

    Trains on mixed queries (with instruction prefix) and documents (without prefix)
    without discrimination. All texts are treated uniformly for pure distillation.

    Supports two modes:
    - With projection: Student (768d) -> Projection -> Teacher (3584d)
    - Without projection (MRL): Student (768d) -> Teacher's first 768d

    Args:
        student_model: Student model (possibly wrapped with DDP)
        teacher_model: Teacher model (frozen)
        config: Training configuration
        device: Device to use
        checkpoint_path: Path to save checkpoints
        rank: Process rank for distributed training (default: 0)
    """
    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("Starting Phase 1: General Distillation")
        logger.info("=" * 80)

    # Get distillation target dimension
    # Extract underlying model if wrapped with DDP
    unwrapped_model = get_ddp_model(student_model)
    target_dim = unwrapped_model.get_output_dim()
    use_projection = unwrapped_model.use_projection

    if is_main_process(rank):
        logger.info(f"Distillation mode: {'With projection' if use_projection else 'MRL-based (no projection)'}")
        logger.info(f"Target dimension: {target_dim}")
        if get_world_size() > 1:
            logger.info(f"Distributed training on {get_world_size()} GPUs")

    # Get tokenizer from student model
    tokenizer = unwrapped_model.student.tokenizer
    max_length = config.get('max_length', 512)

    # Load dataset(s) - REQUIRES pre-computed embeddings
    datasets_config = config.get('datasets')
    precomputed_embeddings_base_dir = config.get('precomputed_embeddings_dir', None)

    if precomputed_embeddings_base_dir is None:
        raise ValueError(
            "Phase 1 requires pre-computed embeddings. Please:\n"
            "1. Add 'precomputed_embeddings_dir' to your config\n"
            "2. Run: torchrun --nproc_per_node=4 precompute_embeddings_v2.py --config configs/mrl.yaml --dataset <dataset_name>"
        )

    if is_main_process(rank):
        logger.info(f"Loading pre-computed embeddings from {precomputed_embeddings_base_dir}")

    # Load each dataset separately and concatenate
    phase1_datasets = []
    for dataset_cfg in datasets_config:
        dataset_name = dataset_cfg['name']
        max_samples = dataset_cfg.get('max_samples', None)

        # Determine precision from directory name (default to fp16)
        # Directory format: {base_dir}/{dataset_name}_{precision}/
        import os
        precision = 'fp16'  # default
        for possible_precision in ['fp16', 'fp32']:
            test_dir = os.path.join(precomputed_embeddings_base_dir, f"{dataset_name}_{possible_precision}")
            if os.path.exists(test_dir):
                precision = possible_precision
                break

        precomputed_dir = os.path.join(precomputed_embeddings_base_dir, f"{dataset_name}_{precision}")

        if is_main_process(rank):
            logger.info(f"Loading dataset: {dataset_name} from {precomputed_dir}")

        ds = Phase1DatasetPrecomputed(
            dataset_name=dataset_name,
            precomputed_embeddings_dir=precomputed_dir,
            max_samples=max_samples,
            data_dir='./beir_datasets',
            dim=target_dim
        )
        phase1_datasets.append(ds)

    # Concatenate all datasets
    if len(phase1_datasets) == 1:
        dataset = phase1_datasets[0]
    else:
        dataset = MultiDatasetWrapper(phase1_datasets)
        if is_main_process(rank):
            logger.info(f"Concatenated {len(phase1_datasets)} datasets: total {len(dataset)} samples")

    if is_main_process(rank):
        logger.info("Using pre-computed teacher embeddings")
        logger.info("Teacher model will not be used for Phase 1")

    # Use DistributedSampler for multi-GPU training
    world_size = get_world_size()
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )


    # Setup optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * config['num_epochs'] // config['gradient_accumulation_steps']
    warmup_steps = config['warmup_steps']

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = Phase1Loss(
        mse_weight=config['mse_weight'],
        cosine_weight=config['cosine_weight']
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    if is_main_process(rank):
        logger.info("Using mixed precision training (fp16)")

    # Training loop
    student_model.train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(config['num_epochs']):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if world_size > 1:
            sampler.set_epoch(epoch)

        if is_main_process(rank):
            logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        epoch_losses = []

        # Only show progress bar on rank 0
        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Phase 1 Epoch {epoch + 1}")
        else:
            pbar = dataloader

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # DataLoader's default collate_fn batches tuples into tuple of batches
            # So batch is (list_of_texts, tensor_of_indices), not list of tuples
            texts, indices = batch
            indices = indices.tolist()  # Convert tensor to list

            # Tokenize inputs
            text_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Get teacher embeddings from pre-computed cache
            with torch.no_grad():
                # Load pre-computed embeddings from RAM to GPU using vectorized indexing
                teacher_emb = torch.from_numpy(dataset.embeddings[indices]).to(device, dtype=torch.float32)

            # Mixed precision forward pass and loss computation
            with torch.amp.autocast('cuda'):
                # Encode with student using forward() for proper gradient flow
                _, student_emb = student_model(
                    text_inputs['input_ids'],
                    text_inputs['attention_mask'],
                    normalize=True
                )

                # Compute loss
                loss, metrics = criterion(student_emb, teacher_emb)
                loss = loss / config['gradient_accumulation_steps']

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_losses.append(loss.item() * config['gradient_accumulation_steps'])

            # Update progress bar (only on rank 0)
            if is_main_process(rank):
                pbar.set_postfix({
                    'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'cosine_sim': f"{metrics['avg_cosine_sim']:.4f}"
                })

        # Gather losses from all processes for accurate averaging
        avg_loss = np.mean(epoch_losses)

        if is_main_process(rank):
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save best model (only on rank 0, with barrier synchronization)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint_dist({
                'epoch': epoch,
                'model_state_dict': get_ddp_model(student_model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{checkpoint_path}.pt", rank)

            if is_main_process(rank):
                logger.info(f"Saved best Phase 1 model with loss: {best_loss:.4f}")

    if is_main_process(rank):
        logger.info("\nPhase 1 training completed!")

    return student_model


def setup_router_trainer(student_model, config, device, rank, total_steps):
    """Setup router and related components for joint training

    Args:
        student_model: Student model (possibly wrapped with DDP)
        config: Training configuration
        device: Device to use
        rank: Process rank for distributed training
        total_steps: Total training steps

    Returns:
        dict: Router trainer components (router, optimizer, loss_fn, weight_scheduler)
    """
    from .models import Router
    from .losses import DifficultyLoss, RouterWeightScheduler

    unwrapped_model = get_ddp_model(student_model)
    student_dim = unwrapped_model.student_dim

    # Initialize router
    router = Router(student_dim=student_dim).to(device)

    # Wrap with DDP if needed
    if get_world_size() > 1:
        router = torch.nn.parallel.DistributedDataParallel(
            router, device_ids=[rank]
        )

    # Optimizer
    router_optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.get('router_lr', 1e-4),
        weight_decay=0.01
    )

    # Loss function
    router_loss_fn = DifficultyLoss(
        infonce_weight=config['infonce_weight'],
        mse_weight=config['mse_weight'],
        temperature=config['temperature']
    )

    # Weight scheduler
    weight_scheduler = RouterWeightScheduler(
        target_weight=config.get('router_loss_weight', 0.1),
        warmup_ratio=config.get('router_warmup_ratio', 0.3),
        total_steps=total_steps
    )

    if is_main_process(rank):
        logger.info("Router training enabled:")
        logger.info(f"  - Target weight: {config.get('router_loss_weight', 0.1)}")
        logger.info(f"  - Warmup ratio: {config.get('router_warmup_ratio', 0.3)}")
        logger.info(f"  - Learning rate: {config.get('router_lr', 1e-4)}")

    return {
        'router': router,
        'optimizer': router_optimizer,
        'loss_fn': router_loss_fn,
        'weight_scheduler': weight_scheduler,
    }


def train_router_step(
    router_trainer,
    student_model,
    query_inputs,
    student_query_emb,
    teacher_query_emb,
    student_doc_embs,
    teacher_doc_embs,
    global_step,
    batch_idx,
    config
):
    """Train router for one step

    Args:
        router_trainer: Dict with router components
        student_model: Student model (possibly wrapped with DDP)
        query_inputs: Tokenized query inputs
        student_query_emb: Student query embeddings
        teacher_query_emb: Teacher query embeddings
        student_doc_embs: Student document embeddings
        teacher_doc_embs: Teacher document embeddings
        global_step: Current global step
        batch_idx: Current batch index
        config: Training configuration

    Returns:
        dict: Router training metrics
    """
    router = router_trainer['router']
    optimizer = router_trainer['optimizer']
    loss_fn = router_trainer['loss_fn']
    weight_scheduler = router_trainer['weight_scheduler']

    # Get current weight
    current_global_step = global_step + (batch_idx + 1) // config['gradient_accumulation_steps']
    router_weight = weight_scheduler.get_weight(current_global_step)

    # Skip if weight is zero
    if router_weight == 0:
        return {'router_weight': 0.0}

    # Get raw student embedding (768d)
    # The forward() method returns (student_emb, output_emb)
    # We need the first value which is always the base 768d embedding
    unwrapped_model = get_ddp_model(student_model)

    with torch.no_grad():
        student_base_emb, _ = unwrapped_model(
            query_inputs['input_ids'],
            query_inputs['attention_mask'],
            normalize=True
        )

    # Predict difficulty
    pred_difficulty = router(student_base_emb.detach())

    # Compute router loss
    router_loss, router_metrics = loss_fn(
        pred_difficulty,
        student_query_emb.detach(),
        teacher_query_emb.detach(),
        student_doc_embs.detach(),
        teacher_doc_embs.detach()
    )

    # Scale loss
    scaled_loss = router_weight * router_loss / config['gradient_accumulation_steps']
    scaled_loss.backward()

    # Update optimizer (gradient accumulation aware)
    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
        torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Return metrics
    router_metrics['router_weight'] = router_weight
    return router_metrics


def train_phase2(student_model, teacher_model, config, device, checkpoint_path, rank=0):
    """Phase 2: Task-specific training with InfoNCE + MSE

    Supports two modes:
    - With projection: Student (768d) -> Projection -> Teacher (3584d)
    - Without projection (MRL): Student (768d) -> Teacher's first 768d

    Args:
        student_model: Student model (possibly wrapped with DDP)
        teacher_model: Teacher model (frozen)
        config: Training configuration
        device: Device to use
        checkpoint_path: Path to save checkpoints
        rank: Process rank for distributed training (default: 0)
    """
    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Starting Phase 2: Task-Specific Training")
        logger.info("=" * 80)

    # Get distillation target dimension
    # Extract underlying model if wrapped with DDP
    unwrapped_model = get_ddp_model(student_model)
    target_dim = unwrapped_model.get_output_dim()
    use_projection = unwrapped_model.use_projection

    if is_main_process(rank):
        logger.info(f"Distillation mode: {'With projection' if use_projection else 'MRL-based (no projection)'}")
        logger.info(f"Target dimension: {target_dim}")
        if get_world_size() > 1:
            logger.info(f"Distributed training on {get_world_size()} GPUs")

    # Get tokenizer from student model
    tokenizer = unwrapped_model.student.tokenizer
    max_length = config.get('max_length', 512)

    # Load dataset(s) - REQUIRES pre-computed embeddings
    datasets_config = config.get('datasets')
    num_negatives = config.get('num_negatives', 7)
    precomputed_embeddings_base_dir = config.get('precomputed_embeddings_dir', None)

    if precomputed_embeddings_base_dir is None:
        raise ValueError(
            "Phase 2 requires pre-computed embeddings. Please:\n"
            "1. Add 'precomputed_embeddings_dir' to your config\n"
            "2. Run: torchrun --nproc_per_node=4 precompute_embeddings_v2.py --config configs/mrl.yaml --dataset <dataset_name>"
        )

    if is_main_process(rank):
        logger.info(f"Loading pre-computed embeddings from {precomputed_embeddings_base_dir}")

    # Load each dataset separately and concatenate
    phase2_datasets = []
    for dataset_cfg in datasets_config:
        dataset_name = dataset_cfg['name']
        max_samples = dataset_cfg.get('max_samples', None)

        # Determine precision from directory name (default to fp16)
        import os
        precision = 'fp16'  # default
        for possible_precision in ['fp16', 'fp32']:
            test_dir = os.path.join(precomputed_embeddings_base_dir, f"{dataset_name}_{possible_precision}")
            if os.path.exists(test_dir):
                precision = possible_precision
                break

        precomputed_dir = os.path.join(precomputed_embeddings_base_dir, f"{dataset_name}_{precision}")

        if is_main_process(rank):
            logger.info(f"Loading dataset: {dataset_name} from {precomputed_dir}")

        ds = Phase2DatasetPrecomputed(
            dataset_name=dataset_name,
            precomputed_embeddings_dir=precomputed_dir,
            num_negatives=num_negatives,
            max_samples=max_samples,
            data_dir='./beir_datasets',
            dim=target_dim
        )
        phase2_datasets.append(ds)

    # Concatenate all datasets
    if len(phase2_datasets) == 1:
        dataset = phase2_datasets[0]
    else:
        dataset = MultiDatasetWrapper(phase2_datasets)
        if is_main_process(rank):
            logger.info(f"Concatenated {len(phase2_datasets)} datasets: total {len(dataset)} samples")

    if is_main_process(rank):
        logger.info("Using pre-computed teacher embeddings")
        logger.info("Teacher model will not be used for Phase 2")

    # Use DistributedSampler for multi-GPU training
    world_size = get_world_size()
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=4,
            collate_fn=phase2_collate_fn,
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=phase2_collate_fn,
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )

    # Setup optimizer with lower learning rate
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    # Learning rate scheduler
    total_steps = len(dataloader) * config['num_epochs'] // config['gradient_accumulation_steps']
    warmup_steps = config['warmup_steps']

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = Phase2Loss(
        infonce_weight=config['infonce_weight'],
        mse_weight=config['mse_weight'],
        temperature=config['temperature']
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    if is_main_process(rank):
        logger.info("Using mixed precision training (fp16)")

    # Optional router setup (only if enabled in config)
    router_trainer = None
    if config.get('train_router', False):
        router_trainer = setup_router_trainer(
            student_model, config, device, rank, total_steps
        )

    # Training loop
    student_model.train()
    if router_trainer is not None:
        router_trainer['router'].train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(config['num_epochs']):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if world_size > 1:
            sampler.set_epoch(epoch)

        if is_main_process(rank):
            logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        epoch_losses = []

        # Only show progress bar on rank 0
        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Phase 2 Epoch {epoch + 1}")
        else:
            pbar = dataloader

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            queries = batch['query']
            positives = batch['positive']
            negatives_list = batch['negatives']  # List of lists (now padded)
            query_indices = batch['_query_indices']
            pos_indices = batch['_pos_indices']
            neg_indices_list = batch['_neg_indices']

            batch_size = len(queries)

            # Prepare documents: [positive, negative1, negative2, ...]
            all_docs = []
            for i in range(batch_size):
                docs = [positives[i]] + negatives_list[i]
                all_docs.append(docs)

            # Tokenize queries
            query_inputs = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Get teacher query embeddings from pre-computed cache using vectorized indexing
            with torch.no_grad():
                # Use vectorized numpy indexing to load all query embeddings at once
                teacher_query_emb = torch.from_numpy(dataset.query_embeddings[query_indices]).to(device, dtype=torch.float32)

            # Mixed precision forward pass for queries
            with torch.amp.autocast('cuda'):
                # Encode queries with student using forward()
                _, student_query_emb = student_model(
                    query_inputs['input_ids'],
                    query_inputs['attention_mask'],
                    normalize=True
                )

            # Encode documents
            teacher_doc_embs = []
            student_doc_embs = []

            for i, docs in enumerate(all_docs):
                # Tokenize documents
                doc_inputs = tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)

                # Get teacher doc embeddings from pre-computed cache using vectorized indexing
                with torch.no_grad():
                    # Load pre-computed embeddings: positive + negatives
                    pos_idx = pos_indices[i]
                    neg_idx_list = neg_indices_list[i]

                    # Use vectorized indexing for positive and negatives
                    pos_emb = dataset.corpus_embeddings[pos_idx:pos_idx+1]  # Keep 2D shape
                    neg_embs = dataset.corpus_embeddings[neg_idx_list]

                    # Combine: [positive, negative1, negative2, ...]
                    doc_embs_np = np.vstack([pos_emb, neg_embs])
                    teacher_docs = torch.from_numpy(doc_embs_np).to(device, dtype=torch.float32)
                    teacher_doc_embs.append(teacher_docs)

                # Encode with student using forward() with mixed precision
                with torch.amp.autocast('cuda'):
                    _, student_docs = student_model(
                        doc_inputs['input_ids'],
                        doc_inputs['attention_mask'],
                        normalize=True
                    )
                student_doc_embs.append(student_docs)

            # Stack to (batch_size, num_docs, dim)
            teacher_doc_embs = torch.stack(teacher_doc_embs)
            student_doc_embs = torch.stack(student_doc_embs)

            # Debug: Check shapes before passing to loss (only once per epoch)
            if batch_idx == 0 and is_main_process(rank):
                logger.info(f"[DEBUG] student_query_emb shape: {student_query_emb.shape}")
                logger.info(f"[DEBUG] student_doc_embs shape: {student_doc_embs.shape}")
                logger.info(f"[DEBUG] batch_size: {batch_size}, num_docs per query: {student_doc_embs.shape[1]}")

            # Compute loss with mixed precision
            with torch.amp.autocast('cuda'):
                loss, metrics = criterion(
                    student_query_emb,
                    teacher_query_emb,
                    student_doc_embs,
                    teacher_doc_embs
                )

            # Optional router training
            if router_trainer is not None:
                router_metrics = train_router_step(
                    router_trainer,
                    student_model,
                    query_inputs,
                    student_query_emb,
                    teacher_query_emb,
                    student_doc_embs,
                    teacher_doc_embs,
                    global_step,
                    batch_idx,
                    config
                )
                metrics.update(router_metrics)

            loss = loss / config['gradient_accumulation_steps']
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_losses.append(loss.item() * config['gradient_accumulation_steps'])

            # Update progress bar (only on rank 0)
            if is_main_process(rank):
                pbar.set_postfix({
                    'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
                    'infonce': f"{metrics['infonce_loss']:.4f}",
                    'mse': f"{metrics['mse_loss']:.6f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # Gather losses from all processes for accurate averaging
        avg_loss = np.mean(epoch_losses)

        if is_main_process(rank):
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save best model (only on rank 0, with barrier synchronization)
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': get_ddp_model(student_model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }

            # Add router state if training router
            if router_trainer is not None:
                checkpoint_dict['router_state_dict'] = \
                    get_ddp_model(router_trainer['router']).state_dict()
                checkpoint_dict['router_optimizer_state_dict'] = \
                    router_trainer['optimizer'].state_dict()

            save_checkpoint_dist(checkpoint_dict, f"{checkpoint_path}.pt", rank)

            if is_main_process(rank):
                logger.info(f"Saved best Phase 2 model with loss: {best_loss:.4f}")

    if is_main_process(rank):
        logger.info("\nPhase 2 training completed!")

    # Return router if it was trained
    if router_trainer is not None:
        return student_model, router_trainer['router']
    return student_model
