#!/usr/bin/env python
"""Pre-compute teacher embeddings for queries and corpus separately.

This script computes teacher embeddings for queries (with instruction prefix) and
corpus separately. This allows reusing embeddings across phases and datasets.

Output structure:
  {base_dir}/{dataset_name}_{precision}/
    ├── queries.mmap       # All queries with instruction prefix
    ├── corpus.mmap        # All corpus documents
    ├── metadata.json      # Metadata with shapes and counts

Usage:
    # Single GPU
    python precompute_embeddings_v2.py --config configs/mrl.yaml --dataset msmarco

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 precompute_embeddings_v2.py --config configs/mrl.yaml --dataset msmarco
"""

import os
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import json

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

from distill import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    TrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][Rank %(rank)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class RankFilter(logging.Filter):
    """Add rank information to log records."""
    def filter(self, record):
        record.rank = get_rank()
        return True


rank_filter = RankFilter()
for handler in logging.root.handlers:
    handler.addFilter(rank_filter)

logger = logging.getLogger(__name__)


def load_beir_dataset(dataset_name, data_dir='./beir_datasets'):
    """Load BEIR dataset and return corpus and queries.

    Returns:
        corpus: dict {doc_id: {'title': ..., 'text': ...}}
        queries: dict {query_id: query_text}
    """
    logger.info(f"Loading BEIR dataset: {dataset_name}")

    # Download and unzip dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = beir_util.download_and_unzip(url, data_dir)

    # Load corpus and queries
    corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="train")

    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")
    return corpus, queries


def encode_texts_sharded(teacher_model, texts, device, batch_size, dtype, rank, world_size):
    """Encode texts with sharding across GPUs.

    Args:
        teacher_model: Teacher model
        texts: List of text strings
        device: CUDA device
        batch_size: Batch size for encoding
        dtype: 'float16' or 'float32'
        rank: Process rank
        world_size: Number of processes

    Returns:
        numpy array of embeddings for this rank's shard
    """
    total_texts = len(texts)
    texts_per_rank = (total_texts + world_size - 1) // world_size
    start_idx = rank * texts_per_rank
    end_idx = min(start_idx + texts_per_rank, total_texts)

    logger.info(f"Rank {rank} encoding texts [{start_idx}:{end_idx}] ({end_idx - start_idx} texts)")

    # Get shard for this rank
    shard_texts = texts[start_idx:end_idx]

    if not shard_texts:
        # Return empty array with correct shape
        sample_emb = teacher_model.encode(["test"], convert_to_tensor=True)
        emb_dim = sample_emb.shape[1]
        np_dtype = np.float16 if dtype == 'float16' else np.float32
        return np.zeros((0, emb_dim), dtype=np_dtype)

    # Encode in batches
    all_embeddings = []
    num_batches = (len(shard_texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=f"Rank {rank}", disable=not is_main_process(rank)):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(shard_texts))
            batch_texts = shard_texts[batch_start:batch_end]

            # Use encode with proper batch_size parameter
            # Note: SentenceTransformer.encode() may internally re-batch
            embeddings = teacher_model.encode(
                batch_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,  # Use the outer batch_size, not len(batch_texts)
                device=device
            )

            # Convert to specified dtype
            if dtype == 'float16':
                embeddings = embeddings.cpu().half().numpy()
            else:
                embeddings = embeddings.cpu().float().numpy()
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, all_embeddings[0].shape[1] if all_embeddings else 768), dtype=np.float16 if dtype == 'float16' else np.float32)


def gather_embeddings_from_all_ranks(local_embeddings, rank, world_size):
    """Gather embeddings from all ranks to rank 0."""
    if world_size == 1:
        return local_embeddings

    # Convert to torch tensor for gathering
    local_tensor = torch.from_numpy(local_embeddings).cuda()

    # Get sizes from all ranks
    local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device='cuda')
    size_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)

    if is_main_process(rank):
        # Rank 0: prepare to receive from all ranks
        size_list = [s.item() for s in size_list]
        max_size = max(size_list) if size_list else 0
        emb_dim = local_tensor.shape[1]

        # Prepare padded tensors for gathering
        gathered_list = []
        for i in range(world_size):
            gathered_list.append(torch.zeros(max_size, emb_dim, dtype=local_tensor.dtype, device='cuda'))
    else:
        gathered_list = None
        size_list = None
        max_size = None

    # Broadcast max_size
    if is_main_process(rank):
        max_size_tensor = torch.tensor([max_size], dtype=torch.long, device='cuda')
    else:
        max_size_tensor = torch.zeros(1, dtype=torch.long, device='cuda')
    torch.distributed.broadcast(max_size_tensor, src=0)
    max_size = max_size_tensor.item()

    # Pad local tensor to max_size
    if local_tensor.shape[0] < max_size:
        padding = torch.zeros(max_size - local_tensor.shape[0], local_tensor.shape[1], dtype=local_tensor.dtype, device='cuda')
        local_tensor_padded = torch.cat([local_tensor, padding], dim=0)
    else:
        local_tensor_padded = local_tensor

    # Gather all tensors at rank 0
    if is_main_process(rank):
        gathered_list[0] = local_tensor_padded
        for i in range(1, world_size):
            torch.distributed.recv(gathered_list[i], src=i)
    else:
        torch.distributed.send(local_tensor_padded, dst=0)

    if is_main_process(rank):
        # Concatenate and trim to actual sizes
        result_list = []
        for i in range(world_size):
            actual_size = size_list[i]
            if actual_size > 0:
                result_list.append(gathered_list[i][:actual_size].cpu().numpy())

        if result_list:
            return np.concatenate(result_list, axis=0)
        else:
            return np.zeros((0, emb_dim), dtype=local_embeddings.dtype)
    else:
        return None


def save_embeddings_memmap(embeddings, output_path, dtype='float16'):
    """Save embeddings as memory-mapped array."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create memmap file
    memmap_array = np.memmap(
        str(output_path),
        dtype=dtype,
        mode='w+',
        shape=embeddings.shape
    )

    # Write data
    memmap_array[:] = embeddings[:]

    # Flush to disk
    memmap_array.flush()
    del memmap_array

    logger.info(f"Saved embeddings to {output_path} with shape {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute teacher embeddings for queries and corpus')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML file')
    parser.add_argument('--dataset', type=str, required=True, help='BEIR dataset name (e.g., msmarco, nfcorpus)')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for encoding (default: 512)')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'], help='Embedding precision (default: fp16)')
    parser.add_argument('--data-dir', type=str, default='./beir_datasets', help='Directory for BEIR datasets')
    parser.add_argument('--max-length', type=int, help="Max length of the texts")
    parser.add_argument('--full-dim', action="store_true", help="Whether to store full dimension of the embedding or store truncated one")
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("Pre-computing Teacher Embeddings (Queries + Corpus)")
        logger.info("=" * 80)
        logger.info(f"Config: {args.config}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Precision: {args.precision}")
        logger.info(f"GPUs: {world_size}")
        logger.info("=" * 80)

    # Load config
    config = TrainingConfig(args.config)

    # Get query instruction
    query_instruction = config.config.get('query_instruction', '')
    if is_main_process(rank):
        logger.info(f"Query instruction: {repr(query_instruction)}")

    # Load teacher model (all ranks)
    if is_main_process(rank):
        logger.info(f"Loading teacher model: {config.model['teacher']}")

    # Determine torch dtype for model loading
    torch_dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    np_dtype = 'float16' if args.precision == 'fp16' else 'float32'

    teacher_model = SentenceTransformer(
        config.model['teacher'],
        trust_remote_code=True,
        model_kwargs={"dtype": torch_dtype}
        # DO NOT use backend="onnx" - it's slow for large models!
    )

    # Set max sequence length (optional - override model default)
    if args.max_length is not None:
        teacher_model.max_seq_length = args.max_length
        if is_main_process(rank):
            logger.info(f"Set max_seq_length to: {args.max_length}")
    else:
        if is_main_process(rank):
            logger.info(f"Using model default max_seq_length: {teacher_model.max_seq_length}")

    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    # Get embedding dimension
    sample_emb = teacher_model.encode(["test"], convert_to_tensor=True, normalize_embeddings=True)
    embedding_dim = sample_emb.shape[1]

    # Determine target dimension
    use_projection = config.model['use_projection']
    if args.full_dim or use_projection:
        target_dim = embedding_dim
    else:
        target_dim = config.model['student_dim']

    if is_main_process(rank):
        logger.info(f"Teacher embedding dimension: {embedding_dim}")
        logger.info(f"Target dimension: {target_dim}")

        # Get output directory
    output_dir = config.get_precomputed_embeddings_dir(dataset_names=[args.dataset], precision=args.precision, dim=None if target_dim==embedding_dim else target_dim)

    if output_dir is None:
        if is_main_process(rank):
            logger.error("Error: 'precomputed_embeddings_dir' not found in config file")
        cleanup_distributed()
        return

    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Load BEIR dataset (all ranks)
    corpus, queries = load_beir_dataset(args.dataset, args.data_dir)

    # Prepare query texts (with instruction prefix)
    query_ids = list(queries.keys())
    query_texts = [query_instruction + queries[qid] for qid in query_ids]

    # Prepare corpus texts
    corpus_ids = list(corpus.keys())
    corpus_texts = []
    for doc_id in corpus_ids:
        doc = corpus[doc_id]
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        corpus_texts.append(text.strip())

    if is_main_process(rank):
        logger.info(f"Total queries: {len(query_texts)}")
        logger.info(f"Total corpus: {len(corpus_texts)}")

    # Encode corpus first (larger dataset)
    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Encoding Corpus")
        logger.info("=" * 80)

    local_corpus_embs = encode_texts_sharded(
        teacher_model, corpus_texts, device, args.batch_size, np_dtype, rank, world_size
    )

    # Slice to target dimension if MRL mode
    if not use_projection and local_corpus_embs.shape[0] > 0:
        local_corpus_embs = local_corpus_embs[:, :target_dim]
        norms = np.linalg.norm(local_corpus_embs, axis=1, keepdims=True)
        local_corpus_embs = (local_corpus_embs / norms).astype(np_dtype)

    # Gather corpus
    if is_main_process(rank):
        logger.info("Gathering corpus embeddings from all GPUs...")
    corpus_embeddings = gather_embeddings_from_all_ranks(local_corpus_embs, rank, world_size)

    # Save corpus embeddings immediately (rank 0 only)
    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Saving Corpus Embeddings")
        logger.info("=" * 80)

        corpus_path = output_dir / "corpus.mmap"
        save_embeddings_memmap(corpus_embeddings, corpus_path, dtype=np_dtype)
        logger.info("✓ Corpus embeddings saved successfully")

        # Free memory
        del corpus_embeddings

    # Encode queries
    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Encoding Queries")
        logger.info("=" * 80)

    local_query_embs = encode_texts_sharded(
        teacher_model, query_texts, device, args.batch_size, np_dtype, rank, world_size
    )

    # Slice to target dimension if MRL mode
    if not use_projection and local_query_embs.shape[0] > 0:
        local_query_embs = local_query_embs[:, :target_dim]
        norms = np.linalg.norm(local_query_embs, axis=1, keepdims=True)
        local_query_embs = (local_query_embs / norms).astype(np_dtype)

    # Gather queries
    if is_main_process(rank):
        logger.info("Gathering query embeddings from all GPUs...")
    query_embeddings = gather_embeddings_from_all_ranks(local_query_embs, rank, world_size)

    # Save query embeddings immediately (rank 0 only)
    if is_main_process(rank):
        logger.info("\n" + "=" * 80)
        logger.info("Saving Query Embeddings")
        logger.info("=" * 80)

        queries_path = output_dir / "queries.mmap"
        if query_embeddings is not None:
            save_embeddings_memmap(query_embeddings, queries_path, dtype=np_dtype)
            logger.info("✓ Query embeddings saved successfully")
            query_shape = query_embeddings.shape
        else:
            raise RuntimeError("Query embeddings is None on rank 0")

        # Save metadata
        corpus_path = output_dir / "corpus.mmap"
        metadata = {
            'dataset': args.dataset,
            'teacher_model': config.model['teacher'],
            'query_instruction': query_instruction,
            'embedding_dim': int(embedding_dim),
            'target_dim': int(target_dim),
            'use_projection': use_projection,
            'dtype': np_dtype,
            'precision': args.precision,
            'num_queries': len(query_ids),
            'num_corpus': len(corpus_ids),
            'query_ids': query_ids,
            'corpus_ids': corpus_ids,
            'query_embeddings_shape': list(query_shape),
            'corpus_embeddings_shape': [len(corpus_ids), target_dim],  # Use expected shape since corpus_embeddings is already freed
            'queries_path': str(queries_path),
            'corpus_path': str(corpus_path),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Pre-computation Complete!")
        logger.info(f"  Corpus: {len(corpus_ids)} documents × {target_dim}d")
        logger.info(f"  Queries: {query_shape[0]} queries × {query_shape[1]}d")
        logger.info("=" * 80)

    cleanup_distributed()


if __name__ == '__main__':
    main()
