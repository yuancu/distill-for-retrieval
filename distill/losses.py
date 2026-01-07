"""Loss functions for distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase1Loss(nn.Module):
    """Phase 1: MSE (0.4) + Cosine (0.6) on normalized embeddings"""
    def __init__(self, mse_weight=0.4, cosine_weight=0.6):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, student_emb, teacher_emb):
        """Both embeddings should be normalized (3584d)"""
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

        # InfoNCE loss (contrastive learning)
        # Compute similarities: (batch_size, num_docs)
        student_sim = torch.matmul(
            query_student.unsqueeze(1),
            doc_student.transpose(1, 2)
        ).squeeze(1) / self.temperature

        # Labels: first document is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_student.device)
        infonce_loss = F.cross_entropy(student_sim, labels)

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
