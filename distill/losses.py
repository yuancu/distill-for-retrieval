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
        """Both embeddings should be normalized"""
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
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG] student_sim shape: {student_sim.shape}, range: [{student_sim.min():.2f}, {student_sim.max():.2f}]")
            print(f"[DEBUG] student_sim[0]: {student_sim[0]}")
        elif not torch.distributed.is_initialized():
            print(f"[DEBUG] student_sim shape: {student_sim.shape}, range: [{student_sim.min():.2f}, {student_sim.max():.2f}]")
            print(f"[DEBUG] student_sim[0]: {student_sim[0]}")

        # Labels: first document is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_student.device)

        # Debug: Check softmax probabilities
        probs = F.softmax(student_sim, dim=-1)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG] Softmax probs[0]: {probs[0]}")
            print(f"[DEBUG] Prob of positive (should be high): {probs[0][0].item():.6f}")
        elif not torch.distributed.is_initialized():
            print(f"[DEBUG] Softmax probs[0]: {probs[0]}")
            print(f"[DEBUG] Prob of positive (should be high): {probs[0][0].item():.6f}")

        infonce_loss = F.cross_entropy(student_sim, labels)

        # Debug: Print InfoNCE loss
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG] infonce_loss: {infonce_loss.item():.6f}")
            print(f"[DEBUG] Expected loss if uniform: {torch.log(torch.tensor(float(num_docs))).item():.6f}")
        elif not torch.distributed.is_initialized():
            print(f"[DEBUG] infonce_loss: {infonce_loss.item():.6f}")
            print(f"[DEBUG] Expected loss if uniform: {torch.log(torch.tensor(float(num_docs))).item():.6f}")

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
