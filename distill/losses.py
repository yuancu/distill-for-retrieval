"""Loss functions for distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase1Loss(nn.Module):
    """Phase 1: MSE + Cosine loss on normalized embeddings

    Note: For normalized embeddings, MSE and cosine are related:
        ||a - b||² = 2(1 - cos(a, b)) when ||a|| = ||b|| = 1

    Using both provides:
    - MSE: Euclidean distance in embedding space
    - Cosine: Direct angular/directional alignment

    For retrieval models using cosine similarity at inference,
    the cosine component is particularly important.
    """
    def __init__(self, mse_weight=0.4, cosine_weight=0.6):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, student_emb, teacher_emb):
        """Both embeddings should be normalized

        Args:
            student_emb: (batch_size, dim) normalized student embeddings
            teacher_emb: (batch_size, dim) normalized teacher embeddings
        """
        # MSE loss on normalized embeddings
        mse_loss = F.mse_loss(student_emb, teacher_emb)

        # Cosine similarity loss (1 - cosine_sim)
        cosine_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()

        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        return total_loss, {
            'loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'avg_cosine_sim': cosine_sim.mean().item()
        }


class Phase2Loss(nn.Module):
    """Phase 2: InfoNCE (0.8) + MSE (0.2)"""
    def __init__(self, infonce_weight=0.8, mse_weight=0.2, temperature=0.02):
        super().__init__()
        self.infonce_weight = infonce_weight
        self.mse_weight = mse_weight
        self.temperature = temperature

    def forward(self, query_student, query_teacher, doc_student, doc_teacher):
        """Compute InfoNCE + MSE loss

        Args:
            query_student: (batch_size, 3584) student query embeddings
            query_teacher: (batch_size, 3584) teacher query embeddings
            doc_student: (batch_size, num_docs, 3584) student doc embeddings
            doc_teacher: (batch_size, num_docs, 3584) teacher doc embeddings
        """
        batch_size = query_student.shape[0]
        num_docs = doc_student.shape[1]

        # Debug: Check shapes
        assert len(query_student.shape) == 2, f"query_student shape should be (batch_size, dim), got {query_student.shape}"
        assert len(doc_student.shape) == 3, f"doc_student shape should be (batch_size, num_docs, dim), got {doc_student.shape}"
        assert num_docs > 1, f"Need at least 2 docs (1 pos + negatives), got {num_docs}"

        # InfoNCE loss (contrastive learning)
        # Compute similarities: (batch_size, num_docs)
        student_sim = torch.matmul(
            query_student.unsqueeze(1),  # (batch_size, 1, dim)
            doc_student.transpose(1, 2)   # (batch_size, dim, num_docs)
        ).squeeze(1) / self.temperature   # (batch_size, num_docs)

        # Debug: Print first batch statistics
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] student_sim shape: {student_sim.shape}, range: [{student_sim.min():.2f}, {student_sim.max():.2f}]")
        #     print(f"[DEBUG] student_sim[0]: {student_sim[0]}")
        # elif not torch.distributed.is_initialized():
        #     print(f"[DEBUG] student_sim shape: {student_sim.shape}, range: [{student_sim.min():.2f}, {student_sim.max():.2f}]")
        #     print(f"[DEBUG] student_sim[0]: {student_sim[0]}")

        # Labels: first document is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_student.device)

        # Debug: Check softmax probabilities
        # probs = F.softmax(student_sim, dim=-1)
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] Softmax probs[0]: {probs[0]}")
        #     print(f"[DEBUG] Prob of positive (should be high): {probs[0][0].item():.6f}")
        # elif not torch.distributed.is_initialized():
        #     print(f"[DEBUG] Softmax probs[0]: {probs[0]}")
        #     print(f"[DEBUG] Prob of positive (should be high): {probs[0][0].item():.6f}")

        infonce_loss = F.cross_entropy(student_sim, labels)

        # Debug: Print InfoNCE loss
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] infonce_loss: {infonce_loss.item():.6f}")
        #     print(f"[DEBUG] Expected loss if uniform: {torch.log(torch.tensor(float(num_docs))).item():.6f}")
        # elif not torch.distributed.is_initialized():
        #     print(f"[DEBUG] infonce_loss: {infonce_loss.item():.6f}")
        #     print(f"[DEBUG] Expected loss if uniform: {torch.log(torch.tensor(float(num_docs))).item():.6f}")

        # MSE loss on query embeddings
        query_mse = F.mse_loss(query_student, query_teacher)

        # MSE loss on document embeddings (average over all docs)
        doc_mse = F.mse_loss(doc_student, doc_teacher)

        mse_loss = (query_mse + doc_mse) / 2

        total_loss = self.infonce_weight * infonce_loss + self.mse_weight * mse_loss

        return total_loss, {
            'loss': total_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'mse_loss': mse_loss.item(),
            'query_mse': query_mse.item(),
            'doc_mse': doc_mse.item()
        }


class DifficultyLoss(nn.Module):
    """Computes difficulty labels and trains router to predict difficulty

    Uses both InfoNCE and MSE losses to compute a continuous difficulty score
    that represents how hard a sample is for the student model.
    """
    def __init__(self, infonce_weight=0.8, mse_weight=0.2, temperature=0.02):
        super().__init__()
        self.infonce_weight = infonce_weight
        self.mse_weight = mse_weight
        self.temperature = temperature

    def compute_difficulty_labels(self, query_student, query_teacher, doc_student, doc_teacher):
        """Compute per-sample difficulty scores from embeddings

        Args:
            query_student: (batch_size, dim) student query embeddings
            query_teacher: (batch_size, dim) teacher query embeddings
            doc_student: (batch_size, num_docs, dim) student doc embeddings
            doc_teacher: (batch_size, num_docs, dim) teacher doc embeddings

        Returns:
            difficulty: (batch_size,) difficulty scores in [0, 1]
            metrics: Dict with difficulty statistics
        """
        batch_size = query_student.shape[0]
        num_docs = doc_student.shape[1]

        with torch.no_grad():
            # Compute per-sample InfoNCE loss (cross entropy without reduction)
            student_sim = torch.matmul(
                query_student.unsqueeze(1),
                doc_student.transpose(1, 2)
            ).squeeze(1) / self.temperature

            labels = torch.zeros(batch_size, dtype=torch.long, device=query_student.device)
            per_sample_infonce = F.cross_entropy(student_sim, labels, reduction='none')

            # Compute per-sample MSE loss on query embeddings
            per_sample_query_mse = ((query_student - query_teacher) ** 2).mean(dim=-1)

            # Normalize InfoNCE loss to [0, 1]
            # Max InfoNCE ≈ log(num_docs), so we normalize by this
            max_infonce = torch.log(torch.tensor(float(num_docs), device=query_student.device))
            normalized_infonce = torch.clamp(per_sample_infonce / max_infonce, 0, 1)

            # Normalize MSE (for normalized embeddings, MSE ∈ [0, 4])
            normalized_mse = torch.clamp(per_sample_query_mse / 4.0, 0, 1)

            # Combined difficulty: weighted average
            difficulty = (
                self.infonce_weight * normalized_infonce +
                self.mse_weight * normalized_mse
            ) / (self.infonce_weight + self.mse_weight)

            metrics = {
                'avg_infonce_difficulty': normalized_infonce.mean().item(),
                'avg_mse_difficulty': normalized_mse.mean().item(),
                'avg_combined_difficulty': difficulty.mean().item(),
            }

        return difficulty, metrics

    def forward(self, pred_difficulty, query_student, query_teacher, doc_student, doc_teacher):
        """Compute loss between predicted and true difficulty

        Args:
            pred_difficulty: (batch_size,) predicted difficulty from router
            query_student: (batch_size, dim) student query embeddings
            query_teacher: (batch_size, dim) teacher query embeddings
            doc_student: (batch_size, num_docs, dim) student doc embeddings
            doc_teacher: (batch_size, num_docs, dim) teacher doc embeddings

        Returns:
            loss: MSE between predicted and true difficulty
            metrics: Dict with difficulty metrics
        """
        # Compute true difficulty labels
        true_difficulty, diff_metrics = self.compute_difficulty_labels(
            query_student, query_teacher, doc_student, doc_teacher
        )

        # Regression loss (MSE)
        loss = F.mse_loss(pred_difficulty, true_difficulty)

        metrics = {
            'router_loss': loss.item(),
            'avg_true_difficulty': true_difficulty.mean().item(),
            'avg_pred_difficulty': pred_difficulty.mean().item(),
            **diff_metrics
        }

        return loss, metrics


class RouterWeightScheduler:
    """Adaptive weight scheduler for router loss

    Schedule:
    - Steps 0 to warmup_steps: weight = 0 (no router training)
    - Steps warmup_steps to total_steps: linear ramp to target_weight
    """
    def __init__(self, target_weight=0.1, warmup_steps=None, total_steps=None, warmup_ratio=0.3):
        """
        Args:
            target_weight: Final weight for router loss (e.g., 0.1)
            warmup_steps: Steps before starting router training (optional)
            total_steps: Total training steps (required)
            warmup_ratio: If warmup_steps not provided, use this ratio of total_steps
        """
        if total_steps is None:
            raise ValueError("total_steps must be provided")

        self.target_weight = target_weight
        self.total_steps = total_steps

        if warmup_steps is None:
            self.warmup_steps = int(total_steps * warmup_ratio)
        else:
            self.warmup_steps = warmup_steps

    def get_weight(self, current_step):
        """Get current weight for router loss

        Args:
            current_step: Current training step

        Returns:
            float: Weight for router loss in [0, target_weight]
        """
        if current_step < self.warmup_steps:
            return 0.0

        # Linear ramp from warmup_steps to total_steps
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]

        return self.target_weight * progress
