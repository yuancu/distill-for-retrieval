"""Model architectures for distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class ProjectionLayer(nn.Module):
    """Projects student embeddings (768d) to teacher dimension (3584d)

    Architecture: 768 → 1536 → 3584 with LayerNorm and GELU activation
    """
    def __init__(self, input_dim=768, hidden_dim=1536, output_dim=3584):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch_size, 768)
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.norm2(x)
        return x  # (batch_size, 3584)


class StudentModelWithProjection(nn.Module):
    """Student model with projection layer for distillation"""
    def __init__(self, student_model_name, projection_layer):
        super().__init__()
        self.student = SentenceTransformer(student_model_name)
        self.projection = projection_layer

    def encode(self, texts, normalize=True, return_projected=True):
        """Encode texts to embeddings

        Args:
            texts: List of strings
            normalize: Whether to L2-normalize embeddings
            return_projected: If True, return 3584d; if False, return 768d
        """
        # Get student embeddings (768d)
        student_emb = self.student.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )

        if not return_projected:
            return student_emb

        # Project to teacher dimension (3584d)
        projected_emb = self.projection(student_emb)

        if normalize:
            projected_emb = F.normalize(projected_emb, p=2, dim=-1)

        return projected_emb

    def forward(self, input_ids, attention_mask, normalize=True):
        """Forward pass for training"""
        # Get student embeddings from base model
        output = self.student[0].auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling
        token_embeddings = output[0]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        student_emb = sum_embeddings / sum_mask

        # Normalize student embeddings
        if normalize:
            student_emb = F.normalize(student_emb, p=2, dim=-1)

        # Project to teacher dimension
        projected_emb = self.projection(student_emb)

        # Normalize projected embeddings
        if normalize:
            projected_emb = F.normalize(projected_emb, p=2, dim=-1)

        return student_emb, projected_emb
