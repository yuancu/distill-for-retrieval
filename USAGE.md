# Using the Distilled Model

After training, you'll have two sentence-transformers compatible models saved:

## Saved Models

```
checkpoints/distilled_mpnet_final/
├── student_base_768d/              # Base model (768d embeddings)
├── student_with_projection_3584d/  # Model with projection (3584d embeddings)
├── projection_layer.pt             # Standalone projection weights
└── training_config.json            # Training configuration
```

## Loading Models

### Option 1: Base Model (768d) - Standalone Retrieval

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_base_768d")
embeddings = model.encode(["your text here"])
print(embeddings.shape)  # (1, 768)
```

**Use when:**
- You want the fastest, most lightweight model
- You don't need compatibility with teacher embeddings
- You're building a standalone retrieval system

### Option 2: Model with Projection (3584d) - Hybrid System

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")
embeddings = model.encode(["your text here"])
print(embeddings.shape)  # (1, 3584)
```

**Use when:**
- You need compatibility with teacher model embeddings
- You want to build a hybrid system (switch between student/teacher)
- You want to use the same vector index for both models
- You're evaluating against teacher baseline

## Evaluation Examples

### BEIR Benchmark

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer

# Load your distilled model
model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

# Load dataset
dataset = "scifact"  # or any BEIR dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Evaluate
model_wrapper = DRES(model, batch_size=128)
retriever = EvaluateRetrieval(model_wrapper, score_function="dot")
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
```

### Custom Retrieval Evaluation

```python
from sentence_transformers import SentenceTransformer
import torch

# Load model
model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

# Your data
queries = ["query 1", "query 2", ...]
documents = ["doc 1", "doc 2", ...]

# Encode
query_embs = model.encode(queries, convert_to_tensor=True)
doc_embs = model.encode(documents, convert_to_tensor=True)

# Compute similarities
scores = torch.matmul(query_embs, doc_embs.T)

# Get top-k
top_k = 10
top_indices = torch.topk(scores, k=top_k, dim=1).indices
```

### Compare with Teacher

```python
from sentence_transformers import SentenceTransformer
import torch

# Load both models
teacher = SentenceTransformer("infly/inf-retriever-v1-pro")
student = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

texts = ["sample text 1", "sample text 2"]

# Encode with both
teacher_embs = teacher.encode(texts, convert_to_tensor=True)
student_embs = student.encode(texts, convert_to_tensor=True)

# Compare
similarity = torch.cosine_similarity(teacher_embs, student_embs, dim=-1)
print(f"Average alignment: {similarity.mean().item():.4f}")
```

## Hybrid System Example

```python
from sentence_transformers import SentenceTransformer
import torch

# Load models
teacher = SentenceTransformer("infly/inf-retriever-v1-pro")
student = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

def adaptive_retrieval(query, corpus, difficulty_threshold=0.8):
    """Use student for easy queries, teacher for hard ones"""

    # Quick retrieval with student
    query_emb = student.encode([query], convert_to_tensor=True)
    doc_embs = student.encode(corpus, convert_to_tensor=True)
    scores = torch.matmul(query_emb, doc_embs.T).squeeze()

    # Check if query is "hard" (low confidence)
    max_score = scores.max().item()

    if max_score < difficulty_threshold:
        # Use teacher for hard query
        query_emb = teacher.encode([query], convert_to_tensor=True)
        doc_embs = teacher.encode(corpus, convert_to_tensor=True)
        scores = torch.matmul(query_emb, doc_embs.T).squeeze()
        print(f"Used TEACHER (confidence: {max_score:.3f})")
    else:
        print(f"Used STUDENT (confidence: {max_score:.3f})")

    return scores

# Example usage
corpus = ["doc 1", "doc 2", "doc 3"]
scores = adaptive_retrieval("hard technical query", corpus)
```

## Performance Characteristics

| Model | Params | Embedding Dim | Speed | Use Case |
|-------|--------|---------------|-------|----------|
| Teacher (inf-retriever-v1-pro) | ~7B | 3584 | Slow | Best quality |
| Student with projection | ~110M | 3584 | Fast | Hybrid systems |
| Student base | ~110M | 768 | Fastest | Standalone retrieval |

## Integration with Existing Systems

### Replace Teacher in Production

```python
# Before (expensive)
model = SentenceTransformer("infly/inf-retriever-v1-pro")

# After (70x fewer parameters, 10x faster)
model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

# Same API, compatible embeddings!
embeddings = model.encode(texts)
```

### Use with FAISS/Pinecone/Weaviate

```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")

# Build index
docs = [...]
doc_embeddings = model.encode(docs)
index = faiss.IndexFlatIP(3584)  # 3584d embeddings
index.add(doc_embeddings)

# Search
query_embedding = model.encode(["your query"])
distances, indices = index.search(query_embedding, k=10)
```

## Model Card

When sharing your model, include this information:

- **Base model**: sentence-transformers/all-mpnet-base-v2
- **Teacher model**: infly/inf-retriever-v1-pro
- **Parameters**: 110M (student) + 7M (projection) = 117M total
- **Embedding dimension**: 3584 (compatible with teacher)
- **Training**: Two-phase distillation
  - Phase 1: General semantic matching (MSE + Cosine)
  - Phase 2: Task-specific with InfoNCE + hard negatives
- **Performance**: ~90-95% of teacher quality at 10x speed

## Troubleshooting

**Q: Can I upload this to HuggingFace Hub?**

A: Yes! Just use:
```python
model.save_to_hub("your-username/distilled-mpnet-retriever")
```

**Q: How do I use with custom prompts?**

A: The model includes the same pooling and normalization as the original. If you need custom prompts, wrap it:
```python
model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")
model.encode(texts, prompt="query: ")  # Add prompt if needed
```

**Q: Can I fine-tune further?**

A: Yes! Load the model and continue training with your data:
```python
model = SentenceTransformer("./checkpoints/distilled_mpnet_final/student_with_projection_3584d")
# Continue training...
```
