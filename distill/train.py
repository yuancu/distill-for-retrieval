"""Training functions for distillation."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

from .datasets import Phase1Dataset, Phase2Dataset, phase2_collate_fn
from .losses import Phase1Loss, Phase2Loss

logger = logging.getLogger(__name__)


def train_phase1(student_model, teacher_model, config, device, checkpoint_path):
    """Phase 1: General distillation with MSE + Cosine loss

    Supports two modes:
    - With projection: Student (768d) -> Projection -> Teacher (3584d)
    - Without projection (MRL): Student (768d) -> Teacher's first 768d
    """
    logger.info("=" * 80)
    logger.info("Starting Phase 1: General Distillation")
    logger.info("=" * 80)

    # Get distillation target dimension
    target_dim = student_model.get_output_dim()
    use_projection = student_model.use_projection
    logger.info(f"Distillation mode: {'With projection' if use_projection else 'MRL-based (no projection)'}")
    logger.info(f"Target dimension: {target_dim}")

    # Get tokenizer from student model
    tokenizer = student_model.student.tokenizer
    max_length = config.get('max_length', 512)

    # Load dataset
    max_samples = config.get('max_samples_per_dataset', 100000)
    dataset = Phase1Dataset(max_samples_per_dataset=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
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

    # Training loop
    student_model.train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Phase 1 Epoch {epoch + 1}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            queries = batch['query']
            passages = batch['passage']

            # Tokenize inputs
            query_inputs = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            passage_inputs = tokenizer(
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Encode with teacher (frozen)
            with torch.no_grad():
                teacher_query_emb = teacher_model.encode(
                    queries,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                ).to(device).clone()  # Clone to convert from inference mode tensor

                teacher_passage_emb = teacher_model.encode(
                    passages,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                ).to(device).clone()  # Clone to convert from inference mode tensor

                # Slice teacher embeddings to target dimension if not using projection (MRL)
                if not use_projection:
                    teacher_query_emb = teacher_query_emb[:, :target_dim]
                    teacher_passage_emb = teacher_passage_emb[:, :target_dim]
                    # Re-normalize after slicing
                    teacher_query_emb = torch.nn.functional.normalize(teacher_query_emb, p=2, dim=-1)
                    teacher_passage_emb = torch.nn.functional.normalize(teacher_passage_emb, p=2, dim=-1)

            # Encode with student using forward() for proper gradient flow
            _, student_query_emb = student_model(
                query_inputs['input_ids'],
                query_inputs['attention_mask'],
                normalize=True
            )

            _, student_passage_emb = student_model(
                passage_inputs['input_ids'],
                passage_inputs['attention_mask'],
                normalize=True
            )

            # Compute loss for both queries and passages
            query_loss, query_metrics = criterion(student_query_emb, teacher_query_emb)
            passage_loss, passage_metrics = criterion(student_passage_emb, teacher_passage_emb)

            loss = (query_loss + passage_loss) / 2
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()

            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_losses.append(loss.item() * config['gradient_accumulation_steps'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'cosine_sim': f"{query_metrics['avg_cosine_sim']:.4f}"
            })

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{checkpoint_path}.pt")
            logger.info(f"Saved best Phase 1 model with loss: {best_loss:.4f}")

    logger.info("\nPhase 1 training completed!")
    return student_model


def train_phase2(student_model, teacher_model, config, device, checkpoint_path):
    """Phase 2: Task-specific training with InfoNCE + MSE

    Supports two modes:
    - With projection: Student (768d) -> Projection -> Teacher (3584d)
    - Without projection (MRL): Student (768d) -> Teacher's first 768d
    """
    logger.info("\n" + "=" * 80)
    logger.info("Starting Phase 2: Task-Specific Training")
    logger.info("=" * 80)

    # Get distillation target dimension
    target_dim = student_model.get_output_dim()
    use_projection = student_model.use_projection
    logger.info(f"Distillation mode: {'With projection' if use_projection else 'MRL-based (no projection)'}")
    logger.info(f"Target dimension: {target_dim}")

    # Get tokenizer from student model
    tokenizer = student_model.student.tokenizer
    max_length = config.get('max_length', 512)

    # Load dataset
    max_samples = config.get('max_samples', 500000)
    dataset = Phase2Dataset(split='train', max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=phase2_collate_fn  # Use custom collate function
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

    # Training loop
    student_model.train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Phase 2 Epoch {epoch + 1}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            queries = batch['query']
            positives = batch['positive']
            negatives_list = batch['negatives']  # List of lists (now padded)

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

            # Encode queries with teacher (frozen)
            with torch.no_grad():
                teacher_query_emb = teacher_model.encode(
                    queries,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                ).to(device).clone()  # Clone to convert from inference mode tensor

                # Slice teacher embeddings to target dimension if not using projection (MRL)
                if not use_projection:
                    teacher_query_emb = teacher_query_emb[:, :target_dim]
                    # Re-normalize after slicing
                    teacher_query_emb = torch.nn.functional.normalize(teacher_query_emb, p=2, dim=-1)

            # Encode queries with student using forward()
            _, student_query_emb = student_model(
                query_inputs['input_ids'],
                query_inputs['attention_mask'],
                normalize=True
            )

            # Encode documents
            teacher_doc_embs = []
            student_doc_embs = []

            for docs in all_docs:
                # Tokenize documents
                doc_inputs = tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)

                # Encode with teacher (frozen)
                with torch.no_grad():
                    teacher_docs = teacher_model.encode(
                        docs,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    ).to(device).clone()  # Clone to convert from inference mode tensor

                    # Slice teacher embeddings to target dimension if not using projection (MRL)
                    if not use_projection:
                        teacher_docs = teacher_docs[:, :target_dim]
                        # Re-normalize after slicing
                        teacher_docs = torch.nn.functional.normalize(teacher_docs, p=2, dim=-1)

                    teacher_doc_embs.append(teacher_docs)

                # Encode with student using forward()
                _, student_docs = student_model(
                    doc_inputs['input_ids'],
                    doc_inputs['attention_mask'],
                    normalize=True
                )
                student_doc_embs.append(student_docs)

            # Stack to (batch_size, num_docs, dim)
            teacher_doc_embs = torch.stack(teacher_doc_embs)
            student_doc_embs = torch.stack(student_doc_embs)

            # Compute loss
            loss, metrics = criterion(
                student_query_emb,
                teacher_query_emb,
                student_doc_embs,
                teacher_doc_embs
            )

            loss = loss / config['gradient_accumulation_steps']
            loss.backward()

            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_losses.append(loss.item() * config['gradient_accumulation_steps'])

            pbar.set_postfix({
                'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
                'infonce': f"{metrics['infonce_loss']:.4f}",
                'mse': f"{metrics['mse_loss']:.6f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{checkpoint_path}.pt")
            logger.info(f"Saved best Phase 2 model with loss: {best_loss:.4f}")

    logger.info("\nPhase 2 training completed!")
    return student_model
