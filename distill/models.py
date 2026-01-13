"""Model architectures for distillation."""

import json
import os
from typing import Dict, Optional

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
    """Student model with optional projection layer for distillation

    Two modes:
    1. With projection: Student (768d) -> Projection -> Teacher dimension (3584d)
    2. Without projection (MRL): Student (768d) -> Teacher's first 768d (MRL-based)
    """
    def __init__(self, student_model_name, projection_layer=None, use_projection=True):
        super().__init__()
        self.student = SentenceTransformer(student_model_name)
        self.projection = projection_layer
        self.use_projection = use_projection

        # Store student dimension
        test_emb = self.student.encode(["test"], convert_to_tensor=True)
        self.student_dim = test_emb.shape[-1]

        if use_projection and projection_layer is None:
            raise ValueError("projection_layer must be provided when use_projection=True")

    def encode(self, texts, normalize=True, return_projected=True):
        """Encode texts to embeddings

        Args:
            texts: List of strings
            normalize: Whether to L2-normalize embeddings
            return_projected: If True and use_projection=True, return projected dim;
                            otherwise return base student embeddings
        """
        # Get student embeddings (e.g., 768d)
        student_emb = self.student.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )

        # If not using projection or not returning projected, return base embeddings
        if not self.use_projection or not return_projected:
            return student_emb

        # Project to teacher dimension (e.g., 3584d)
        projected_emb = self.projection(student_emb)

        if normalize:
            projected_emb = F.normalize(projected_emb, p=2, dim=-1)

        return projected_emb

    def forward(self, input_ids, attention_mask, normalize=True):
        """Forward pass for training

        Returns:
            student_emb: Base student embeddings (e.g., 768d)
            output_emb: Either projected embeddings (if use_projection) or student_emb
        """
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

        # If not using projection, return student embeddings for both
        if not self.use_projection:
            return student_emb, student_emb

        # Project to teacher dimension
        projected_emb = self.projection(student_emb)

        # Normalize projected embeddings
        if normalize:
            projected_emb = F.normalize(projected_emb, p=2, dim=-1)

        return student_emb, projected_emb

    def get_output_dim(self):
        """Get the output dimension for distillation

        Returns:
            int: Output dimension (student_dim if not using projection, else projected dim)
        """
        if not self.use_projection:
            return self.student_dim
        # Get projection output dimension
        return self.projection.layer2.out_features


class Router(nn.Module):
    """Routes queries to student or teacher model based on predicted difficulty

    At inference time, uses only student embedding to predict if the query is
    too difficult for the student model. If difficulty exceeds threshold,
    routes to teacher model for better quality embeddings.

    Architecture: 768d student embedding -> MLP -> difficulty score [0, 1]
    """
    def __init__(self, student_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, student_query_emb):
        """Predict difficulty score for routing decision

        Args:
            student_query_emb: (batch_size, student_dim) - ONLY student embeddings

        Returns:
            difficulty_scores: (batch_size,) - predicted difficulty in [0, 1]
                              Higher score = more difficult = should route to teacher
        """
        return self.net(student_query_emb).squeeze(-1)


class RoutingModule(nn.Module):
    """Routes between student and teacher embeddings based on predicted difficulty.

    This module can be used in a SentenceTransformer pipeline:
    Transformer → Pooling → RoutingModule → Normalize

    The module:
    1. Receives student embeddings from previous modules
    2. Uses router to predict query difficulty
    3. For difficult queries (difficulty > threshold):
       - Decodes input_ids back to text
       - Encodes with teacher model
       - Returns teacher embeddings (truncated to student dim)
    4. For easy queries: returns student embeddings as-is
    5. Stores routing decisions in features['routing_decisions']
    """

    def __init__(
        self,
        teacher_model: SentenceTransformer,
        router: Router,
        threshold: float,
        student_dim: int,
        router_hidden_dim: int = 256
    ):
        """Initialize routing module.

        Args:
            teacher_model: Teacher SentenceTransformer model
            router: Trained Router for difficulty prediction
            threshold: Routing threshold (0-1). If difficulty > threshold, use teacher
            student_dim: Dimension of student embeddings
            router_hidden_dim: Hidden dimension of router (for saving/loading)
        """
        super().__init__()

        # Store components (teacher and router are part of this module's state)
        self.teacher_model = teacher_model
        self.router = router
        self.threshold = threshold
        self.student_dim = student_dim
        self.router_hidden_dim = router_hidden_dim

        # Config keys for SentenceTransformer serialization
        self.config_keys = ['threshold', 'student_dim', 'router_hidden_dim']

        # Set to eval mode
        self.teacher_model.eval()
        self.router.eval()

        # Store last routing decisions for access after encoding
        self.last_routing_decisions = None

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply routing logic to embeddings.

        Args:
            features: Dict containing:
                - 'sentence_embedding': Student embeddings from previous modules
                - 'input_ids': Tokenized input (for decoding back to text)
                - 'attention_mask': Attention mask
                - Optionally 'token_embeddings' from Transformer

        Returns:
            Dict with updated 'sentence_embedding' (student or teacher)
            and 'routing_decisions' (boolean tensor)
        """
        student_emb = features['sentence_embedding']  # (batch_size, student_dim)
        batch_size = student_emb.shape[0]

        # Handle case where embeddings are already truncated or have different dim
        # For router, we need exactly student_dim dimensions
        if student_emb.shape[1] != self.student_dim:
            raise ValueError(f"Student embedding mismatch. Expected: {self.student_dim}, actual: {student_emb.shape[1]}")
        router_input = student_emb

        # Predict difficulty with router (expects unnormalized embeddings)
        with torch.no_grad():
            difficulty_scores = self.router(router_input.detach())  # (batch_size,)

        # Determine routing (True = use teacher)
        use_teacher = difficulty_scores > self.threshold

        # Store routing decisions for access after encoding
        self.last_routing_decisions = use_teacher.cpu()

        # If no queries need teacher, return student embeddings as-is
        if not use_teacher.any():
            features['routing_decisions'] = use_teacher
            return features

        # For queries routed to teacher, we need to re-encode with teacher
        # Decode input_ids back to text using the tokenizer
        if 'input_ids' not in features:
            # Fallback: if no input_ids, can't re-encode with teacher
            # Just return student embeddings
            features['routing_decisions'] = torch.zeros(batch_size, dtype=torch.bool, device=student_emb.device)
            return features

        # Get tokenizer from teacher model
        tokenizer = self.teacher_model.tokenizer

        # Decode input_ids back to text
        input_ids = features['input_ids']
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Get indices of sentences to route to teacher
        teacher_indices = torch.where(use_teacher)[0]
        teacher_texts = [texts[i] for i in teacher_indices.cpu().numpy()]

        # Encode with teacher
        with torch.no_grad():
            teacher_embs = self.teacher_model.encode(
                teacher_texts,
                convert_to_tensor=True,
                normalize_embeddings=False,  # Normalization happens in next module
                device=student_emb.device,
                show_progress_bar=False
            )

        # Truncate teacher embeddings to student_dim
        teacher_embs = teacher_embs[:, :self.student_dim]

        # Replace student embeddings with teacher embeddings for difficult queries
        features['sentence_embedding'][teacher_indices] = teacher_embs

        # Add routing decisions to features
        features['routing_decisions'] = use_teacher

        return features

    def get_sentence_embedding_dimension(self) -> Optional[int]:
        """Return output dimension (unchanged from input)."""
        return None  # Doesn't change dimension

    def save(self, output_path: str):
        """Save the routing module (teacher model, router, config)."""
        os.makedirs(output_path, exist_ok=True)

        # Save teacher model to subdirectory
        teacher_path = os.path.join(output_path, 'teacher')
        self.teacher_model.save(teacher_path)

        # Save router weights
        router_path = os.path.join(output_path, 'router.pt')
        torch.save(self.router.state_dict(), router_path)

        # Save config
        config = {
            'threshold': float(self.threshold),
            'student_dim': int(self.student_dim),
            'router_hidden_dim': int(self.router_hidden_dim),
        }

        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def load(input_path: str):
        """Load the routing module from directory.

        Args:
            input_path: Path to saved routing module

        Returns:
            RoutingModule instance
        """
        # Load config
        config_path = os.path.join(input_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load teacher model
        teacher_path = os.path.join(input_path, 'teacher')
        teacher_model = SentenceTransformer(teacher_path)

        # Load router
        router = Router(
            student_dim=config['student_dim'],
            hidden_dim=config['router_hidden_dim']
        )
        router_path = os.path.join(input_path, 'router.pt')
        state_dict = torch.load(router_path, map_location='cpu', weights_only=True)
        router.load_state_dict(state_dict)

        # Create and return module
        module = RoutingModule(
            teacher_model=teacher_model,
            router=router,
            threshold=config['threshold'],
            student_dim=config['student_dim'],
            router_hidden_dim=config['router_hidden_dim']
        )

        return module
