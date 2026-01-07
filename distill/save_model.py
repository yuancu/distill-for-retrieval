"""Save distilled model with projection as a SentenceTransformer model to artifacts/"""

import os
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from typing import Dict, Union, Any, Optional

# Import the projection layer from distill module
from distill import ProjectionLayer, StudentModelWithProjection


class ProjectionModule(nn.Module):
    """Custom SentenceTransformer module that wraps the trained projection layer

    This module integrates seamlessly with SentenceTransformer's pipeline:
    Transformer → Pooling → ProjectionModule → Normalize
    """

    def __init__(self, projection_layer: ProjectionLayer):
        super().__init__()
        self.projection_layer = projection_layer

        # Store config for saving/loading
        self.config_keys = ['in_features', 'hidden_features', 'out_features']
        self.in_features = projection_layer.layer1.in_features
        self.hidden_features = projection_layer.layer1.out_features
        self.out_features = projection_layer.layer2.out_features

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply projection to sentence embeddings

        Args:
            features: Dict with 'sentence_embedding' key containing embeddings

        Returns:
            Dict with projected 'sentence_embedding'
        """
        # Apply the projection layer to sentence embeddings
        features['sentence_embedding'] = self.projection_layer(features['sentence_embedding'])
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Return the output dimension after projection"""
        return self.out_features

    def save(self, output_path: str):
        """Save the projection module"""
        os.makedirs(output_path, exist_ok=True)

        # Save the projection layer weights
        torch.save(self.projection_layer.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

        # Save config
        config = {
            'in_features': self.in_features,
            'hidden_features': self.hidden_features,
            'out_features': self.out_features,
        }
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Save model.py to the parent directory (root of the model)
        parent_dir = os.path.dirname(output_path)
        write_model_py(parent_dir)

    @staticmethod
    def load(input_path: str):
        """Load the projection module"""
        # Load config
        with open(os.path.join(input_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create projection layer
        projection_layer = ProjectionLayer(
            input_dim=config['in_features'],
            hidden_dim=config['hidden_features'],
            output_dim=config['out_features']
        )

        # Load weights
        state_dict = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        projection_layer.load_state_dict(state_dict)

        # Create and return module
        return ProjectionModule(projection_layer)


def write_model_py(output_dir: str):
    """Write a self-contained model.py file to the output directory

    This creates a standalone model.py that includes both ProjectionLayer
    and ProjectionModule classes without external dependencies (except PyTorch).
    """
    model_py_content = '''"""Distilled model with projection layer for SentenceTransformer"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict


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


class ProjectionModule(nn.Module):
    """Custom SentenceTransformer module that wraps the trained projection layer

    This module integrates seamlessly with SentenceTransformer's pipeline:
    Transformer → Pooling → ProjectionModule → Normalize
    """

    def __init__(self, projection_layer: ProjectionLayer):
        super().__init__()
        self.projection_layer = projection_layer

        # Store config for saving/loading
        self.config_keys = ['in_features', 'hidden_features', 'out_features']
        self.in_features = projection_layer.layer1.in_features
        self.hidden_features = projection_layer.layer1.out_features
        self.out_features = projection_layer.layer2.out_features

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply projection to sentence embeddings

        Args:
            features: Dict with 'sentence_embedding' key containing embeddings

        Returns:
            Dict with projected 'sentence_embedding'
        """
        # Apply the projection layer to sentence embeddings
        features['sentence_embedding'] = self.projection_layer(features['sentence_embedding'])
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Return the output dimension after projection"""
        return self.out_features

    def save(self, output_path: str):
        """Save the projection module"""
        os.makedirs(output_path, exist_ok=True)

        # Save the projection layer weights
        torch.save(self.projection_layer.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

        # Save config
        config = {
            'in_features': self.in_features,
            'hidden_features': self.hidden_features,
            'out_features': self.out_features,
        }
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def load(input_path: str):
        """Load the projection module"""
        # Load config
        with open(os.path.join(input_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create projection layer
        projection_layer = ProjectionLayer(
            input_dim=config['in_features'],
            hidden_dim=config['hidden_features'],
            output_dim=config['out_features']
        )

        # Load weights
        state_dict = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        projection_layer.load_state_dict(state_dict)

        # Create and return module
        return ProjectionModule(projection_layer)
'''

    model_py_path = os.path.join(output_dir, 'model.py')
    with open(model_py_path, 'w') as f:
        f.write(model_py_content)

    print(f"   ✓ Saved self-contained model.py to {model_py_path}")


def save_distilled_model_to_artifacts(
    student_model: StudentModelWithProjection,
    checkpoint_path: str,
    artifacts_dir: str = "./artifacts",
    model_name: Optional[str] = None
):
    """
    Save the distilled student model as a SentenceTransformer

    Supports two modes:
    - With projection: Saves model with projection layer (e.g., 768d -> 3584d)
    - Without projection (MRL): Saves base student model only (e.g., 768d)

    Args:
        student_model: The trained StudentModelWithProjection
        checkpoint_path: Path to the checkpoint (.pt file) to load weights from
        artifacts_dir: Directory to save the model (default: ./artifacts)
        model_name: Name for the saved model directory (auto-generated if None)
    """

    print("=" * 80)
    print("Saving Distilled Model to Artifacts")
    print("=" * 80)

    # Load the best checkpoint
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()
    print("   ✓ Checkpoint loaded successfully")

    # Determine mode and output dimension
    use_projection = student_model.use_projection
    output_dim = student_model.get_output_dim()
    mode_str = "WITH PROJECTION" if use_projection else "MRL-BASED (no projection)"
    print(f"\n   Mode: {mode_str}")
    print(f"   Output dimension: {output_dim}d")

    # Auto-generate model name if not provided
    if model_name is None:
        model_name = f"distilled-mpnet-{output_dim}d"
        if not use_projection:
            model_name += "-mrl"

    # Create output directory
    output_path = os.path.join(artifacts_dir, model_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"\n2. Creating output directory: {output_path}")

    # Extract components from the student model
    print("\n3. Extracting model components...")

    # Get the transformer and pooling modules from the student SentenceTransformer
    transformer = student_model.student[0]  # Transformer module
    pooling = student_model.student[1]      # Pooling module
    print(f"   ✓ Transformer: {transformer}")
    print(f"   ✓ Pooling: {pooling}")

    # Build model modules based on mode
    modules = [transformer, pooling]

    if use_projection:
        # Create custom projection module
        projection = ProjectionModule(student_model.projection)
        print(f"   ✓ Projection: {projection.in_features}d → {projection.hidden_features}d → {projection.out_features}d")
        modules.append(projection)
    else:
        print(f"   ✓ No projection (MRL mode)")

    # Add normalization module
    normalize = models.Normalize()
    print(f"   ✓ Normalization: L2 normalization")
    modules.append(normalize)

    # Build the complete SentenceTransformer model
    print("\n4. Building SentenceTransformer model...")
    model = SentenceTransformer(modules=modules)
    print("   ✓ Model assembled successfully")

    # Save the model
    print(f"\n5. Saving model to: {output_path}")
    model.save(output_path)
    print("   ✓ Model saved successfully")

    # Save additional metadata
    print("\n6. Saving metadata...")

    # Build metadata based on mode
    metadata = {
        "model_type": "distilled-retrieval-model",
        "base_model": "sentence-transformers/all-mpnet-base-v2",
        "teacher_model": "infly/inf-retriever-v1-pro",
        "embedding_dimension": output_dim,
        "base_dimension": student_model.student_dim,
        "distillation_mode": "projection" if use_projection else "mrl",
        "usage": "model = SentenceTransformer('{}')".format(output_path)
    }

    if use_projection:
        metadata.update({
            "projection_hidden_dim": student_model.projection.layer1.out_features,
            "architecture": "MPNet + 2-Layer MLP Projection",
            "description": f"Distilled retrieval model with projection to {output_dim}d (teacher-compatible)"
        })
    else:
        metadata.update({
            "architecture": "MPNet (base model only)",
            "description": f"Distilled retrieval model using MRL approach ({output_dim}d, no projection)"
        })

    with open(os.path.join(output_path, 'model_card.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✓ Metadata saved")

    # Test the saved model
    print("\n7. Testing saved model...")
    loaded_model = SentenceTransformer(output_path)
    test_texts = ["This is a test sentence."]
    embeddings = loaded_model.encode(test_texts, convert_to_tensor=True)
    print(f"   ✓ Test encoding successful: {embeddings.shape}")
    assert embeddings.shape[1] == output_dim, f"Expected {output_dim}d embeddings, got {embeddings.shape[1]}d"
    print(f"   ✓ Embedding dimension verified: {output_dim}d")

    # Print summary
    print("\n" + "=" * 80)
    print("✓ Model saved successfully!")
    print("=" * 80)
    print(f"\nModel location: {output_path}")
    print(f"Distillation mode: {mode_str}")
    print(f"Embedding dimension: {output_dim}d")
    print("\nTo use this model:")
    print("```python")
    print("from sentence_transformers import SentenceTransformer")
    print(f"model = SentenceTransformer('{output_path}')")
    print("embeddings = model.encode(['your', 'texts', 'here'])")
    print("```")
    print("\nThe model will automatically apply:")
    print("  1. Tokenization (MPNet tokenizer)")
    print(f"  2. Encoding (MPNet base model → {student_model.student_dim}d)")
    print("  3. Pooling (Mean pooling)")

    if use_projection:
        proj_hidden = student_model.projection.layer1.out_features
        print(f"  4. Projection ({student_model.student_dim}d → {proj_hidden}d → {output_dim}d)")
        print("  5. Normalization (L2 normalization)")
    else:
        print("  4. Normalization (L2 normalization)")
        print(f"\nNote: This model uses MRL distillation (no projection).")
        print(f"It's trained to match the teacher's first {output_dim} dimensions.")

    return output_path


if __name__ == "__main__":
    # Configuration
    TEACHER_MODEL = "infly/inf-retriever-v1-pro"
    STUDENT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TEACHER_DIM = 3584
    STUDENT_DIM = 768
    PROJECTION_HIDDEN_DIM = 1536

    # Distillation mode (set to False for MRL mode)
    USE_PROJECTION = True

    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2_best.pt")
    ARTIFACTS_DIR = "./artifacts"

    # Check if checkpoint exists
    if not os.path.exists(PHASE2_CHECKPOINT):
        print(f"Error: Checkpoint not found at {PHASE2_CHECKPOINT}")
        print("Please run the training first to generate the checkpoint.")
        exit(1)

    # Create student model based on mode
    print("Loading student model...")
    print(f"Mode: {'WITH PROJECTION' if USE_PROJECTION else 'MRL-BASED (no projection)'}")

    if USE_PROJECTION:
        projection_layer = ProjectionLayer(STUDENT_DIM, PROJECTION_HIDDEN_DIM, TEACHER_DIM)
        student_model = StudentModelWithProjection(
            STUDENT_MODEL,
            projection_layer=projection_layer,
            use_projection=True
        )
    else:
        student_model = StudentModelWithProjection(
            STUDENT_MODEL,
            projection_layer=None,
            use_projection=False
        )

    # Save to artifacts (model_name is auto-generated based on mode)
    output_path = save_distilled_model_to_artifacts(
        student_model=student_model,
        checkpoint_path=PHASE2_CHECKPOINT,
        artifacts_dir=ARTIFACTS_DIR,
        model_name=None  # Auto-generate based on mode
    )

    print(f"\n✓ Done! Model saved to: {output_path}")
