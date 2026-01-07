"""Evaluation functions for distillation."""

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


def evaluate_retrieval(student_model, teacher_model, device, num_samples=1000):
    """Evaluate student vs teacher on MS MARCO dev set"""
    logger.info("\n" + "=" * 80)
    logger.info("Intermediate Evaluation")
    logger.info("=" * 80)

    # Load MS MARCO dev set
    dev_dataset = load_dataset('ms_marco', 'v1.1', split='validation')

    student_model.eval()
    teacher_model.eval()

    cosine_sims = []
    mse_errors = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(dev_dataset, total=num_samples, desc="Evaluating")):
            if i >= num_samples:
                break

            query = item['query']

            # Find positive passage
            positive_idx = None
            for idx, is_selected in enumerate(item['passages']['is_selected']):
                if is_selected:
                    positive_idx = idx
                    break

            if positive_idx is None:
                continue

            passage = item['passages']['passage_text'][positive_idx]

            # Encode with both models
            teacher_query = teacher_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            teacher_passage = teacher_model.encode([passage], convert_to_tensor=True, normalize_embeddings=True)

            student_query = student_model.encode([query], normalize=True, return_projected=True)
            student_passage = student_model.encode([passage], normalize=True, return_projected=True)

            # Compute metrics
            query_cosine = F.cosine_similarity(student_query, teacher_query).item()
            passage_cosine = F.cosine_similarity(student_passage, teacher_passage).item()

            query_mse = F.mse_loss(student_query, teacher_query).item()
            passage_mse = F.mse_loss(student_passage, teacher_passage).item()

            cosine_sims.append((query_cosine + passage_cosine) / 2)
            mse_errors.append((query_mse + passage_mse) / 2)

    # Report metrics
    avg_cosine = np.mean(cosine_sims)
    avg_mse = np.mean(mse_errors)

    logger.info(f"\nEvaluation Results ({num_samples} samples):")
    logger.info(f"  Average Cosine Similarity: {avg_cosine:.4f}")
    logger.info(f"  Average MSE: {avg_mse:.6f}")
    logger.info(f"  Embedding alignment: {'Good' if avg_cosine > 0.9 else 'Needs improvement'}")

    student_model.train()
    return avg_cosine, avg_mse
