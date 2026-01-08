"""Simplified dataset classes for pre-computed embeddings.

These datasets load pre-computed query and corpus embeddings and build
training samples on-demand.
"""

import logging
import json
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


class Phase1DatasetPrecomputed(Dataset):
    """Phase 1 dataset using pre-computed query and corpus embeddings.

    Loads queries.mmap and corpus.mmap, concatenates them in order
    (queries first, then corpus). No shuffling or sampling ratio.

    Args:
        dataset_name: BEIR dataset name (e.g., 'msmarco')
        precomputed_embeddings_dir: Directory containing queries.mmap and corpus.mmap
        max_samples: Maximum number of samples (optional, truncates if set)
        data_dir: Directory for BEIR datasets (to load text samples)
    """

    def __init__(self, dataset_name, precomputed_embeddings_dir, max_samples=None, data_dir='./beir_datasets'):
        self.dataset_name = dataset_name
        self.precomputed_embeddings_dir = Path(precomputed_embeddings_dir)
        self.max_samples = max_samples
        self.data_dir = data_dir

        # Load metadata
        metadata_path = self.precomputed_embeddings_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load BEIR dataset for text samples
        logger.info(f"Loading BEIR dataset: {dataset_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, data_dir)
        corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="train")

        # Get query instruction from metadata
        query_instruction = self.metadata.get('query_instruction', '')

        # Build samples list: queries first, then corpus (in order)
        self.samples = []
        self.embedding_indices = []

        # Add queries
        query_ids = self.metadata['query_ids']
        for i, qid in enumerate(query_ids):
            if qid in queries:
                text = query_instruction + queries[qid]
                self.samples.append(text)
                self.embedding_indices.append(i)  # Index in queries.mmap

        num_queries = len(self.samples)

        # Add corpus
        corpus_ids = self.metadata['corpus_ids']
        for i, doc_id in enumerate(corpus_ids):
            if doc_id in corpus:
                doc = corpus[doc_id]
                text = doc.get('title', '') + ' ' + doc.get('text', '')
                self.samples.append(text.strip())
                # Index in concatenated embeddings (queries + corpus)
                self.embedding_indices.append(num_queries + i)

        logger.info(f"Loaded {num_queries} queries and {len(corpus_ids)} corpus ({len(self.samples)} total)")

        # Truncate if max_samples is set
        if max_samples is not None and max_samples < len(self.samples):
            logger.info(f"Truncating to {max_samples} samples")
            self.samples = self.samples[:max_samples]
            self.embedding_indices = self.embedding_indices[:max_samples]

        # Load embeddings into RAM
        self._load_embeddings()

    def _load_embeddings(self):
        """Load pre-computed embeddings into RAM."""
        dtype_str = self.metadata.get('dtype', 'float16')

        # Load queries
        query_shape = tuple(self.metadata['query_embeddings_shape'])
        query_path = self.precomputed_embeddings_dir / "queries.mmap"
        query_memmap = np.memmap(str(query_path), dtype=dtype_str, mode='r', shape=query_shape)
        query_embeddings = np.array(query_memmap)

        # Load corpus
        corpus_shape = tuple(self.metadata['corpus_embeddings_shape'])
        corpus_path = self.precomputed_embeddings_dir / "corpus.mmap"
        corpus_memmap = np.memmap(str(corpus_path), dtype=dtype_str, mode='r', shape=corpus_shape)
        corpus_embeddings = np.array(corpus_memmap)

        # Concatenate: queries first, then corpus
        self.embeddings = np.concatenate([query_embeddings, corpus_embeddings], axis=0)

        # Truncate embeddings if needed
        if self.max_samples is not None and self.max_samples < self.embeddings.shape[0]:
            self.embeddings = self.embeddings[:self.max_samples]

        self.embedding_dim = self.embeddings.shape[1]

        bytes_per_elem = 2 if dtype_str == 'float16' else 4
        size_mb = (self.embeddings.shape[0] * self.embeddings.shape[1] * bytes_per_elem) / (1024 * 1024)
        logger.info(f"Loaded embeddings into RAM: {self.embeddings.shape} ({size_mb:.2f} MB)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return (text, embedding_index)."""
        text = self.samples[idx]
        emb_idx = self.embedding_indices[idx]
        return text, emb_idx

    def get_embedding(self, idx):
        """Get embedding by sample index."""
        emb_idx = self.embedding_indices[idx]
        return self.embeddings[emb_idx]


class Phase2DatasetPrecomputed(Dataset):
    """Phase 2 dataset using pre-computed query and corpus embeddings.

    Builds query-document triplets on-demand from qrels using pre-computed embeddings.

    Args:
        dataset_name: BEIR dataset name
        precomputed_embeddings_dir: Directory containing queries.mmap and corpus.mmap
        num_negatives: Number of hard negatives per query
        max_samples: Maximum number of query samples
        data_dir: Directory for BEIR datasets
    """

    def __init__(self, dataset_name, precomputed_embeddings_dir, num_negatives=7,
                 max_samples=None, data_dir='./beir_datasets'):
        self.dataset_name = dataset_name
        self.precomputed_embeddings_dir = Path(precomputed_embeddings_dir)
        self.num_negatives = num_negatives
        self.max_samples = max_samples
        self.data_dir = data_dir

        # Load metadata
        metadata_path = self.precomputed_embeddings_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load BEIR dataset
        logger.info(f"Loading BEIR dataset: {dataset_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, data_dir)
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_folder=data_path).load(split="train")

        # Build ID to index mappings
        self.query_ids = self.metadata['query_ids']
        self.corpus_ids = self.metadata['corpus_ids']
        self.query_id_to_idx = {qid: i for i, qid in enumerate(self.query_ids)}
        self.corpus_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}

        # Build training samples from qrels
        self._build_samples()

        # Load embeddings
        self._load_embeddings()

    def _build_samples(self):
        """Build training samples from qrels."""
        self.samples = []

        for query_id, doc_scores in self.qrels.items():
            if query_id not in self.queries or query_id not in self.query_id_to_idx:
                continue

            # Separate positive and negative documents
            positive_docs = [(doc_id, score) for doc_id, score in doc_scores.items() if score > 0]
            negative_docs = [(doc_id, score) for doc_id, score in doc_scores.items() if score == 0]

            if not positive_docs:
                continue

            # Use highest scoring positive document
            pos_doc_id, _ = max(positive_docs, key=lambda x: x[1])
            if pos_doc_id not in self.corpus or pos_doc_id not in self.corpus_id_to_idx:
                continue

            # Collect hard negatives
            negative_doc_ids = []
            for neg_doc_id, _ in negative_docs:
                if len(negative_doc_ids) >= self.num_negatives:
                    break
                if neg_doc_id in self.corpus and neg_doc_id in self.corpus_id_to_idx:
                    negative_doc_ids.append(neg_doc_id)

            # Sample random negatives if needed
            if len(negative_doc_ids) < self.num_negatives:
                judged_doc_ids = set(doc_scores.keys())
                available_corpus_ids = [
                    doc_id for doc_id in self.corpus_ids
                    if doc_id not in judged_doc_ids and doc_id in self.corpus
                ]
                num_needed = self.num_negatives - len(negative_doc_ids)
                if len(available_corpus_ids) >= num_needed:
                    sampled = random.sample(available_corpus_ids, num_needed)
                    negative_doc_ids.extend(sampled)

            # Only add if we have at least one negative
            if negative_doc_ids:
                self.samples.append({
                    'query_id': query_id,
                    'pos_doc_id': pos_doc_id,
                    'neg_doc_ids': negative_doc_ids[:self.num_negatives]
                })

            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        logger.info(f"Built {len(self.samples)} Phase 2 training samples")

    def _load_embeddings(self):
        """Load pre-computed embeddings into RAM."""
        dtype_str = self.metadata.get('dtype', 'float16')

        # Load queries
        query_shape = tuple(self.metadata['query_embeddings_shape'])
        query_path = self.precomputed_embeddings_dir / "queries.mmap"
        query_memmap = np.memmap(str(query_path), dtype=dtype_str, mode='r', shape=query_shape)
        self.query_embeddings = np.array(query_memmap)

        # Load corpus
        corpus_shape = tuple(self.metadata['corpus_embeddings_shape'])
        corpus_path = self.precomputed_embeddings_dir / "corpus.mmap"
        corpus_memmap = np.memmap(str(corpus_path), dtype=dtype_str, mode='r', shape=corpus_shape)
        self.corpus_embeddings = np.array(corpus_memmap)

        self.embedding_dim = query_shape[1]

        bytes_per_elem = 2 if dtype_str == 'float16' else 4
        query_mb = (query_shape[0] * query_shape[1] * bytes_per_elem) / (1024 * 1024)
        corpus_mb = (corpus_shape[0] * corpus_shape[1] * bytes_per_elem) / (1024 * 1024)

        logger.info(f"Loaded embeddings into RAM:")
        logger.info(f"  Queries: {query_shape} ({query_mb:.2f} MB)")
        logger.info(f"  Corpus: {corpus_shape} ({corpus_mb:.2f} MB)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return sample dict with texts and _emb_idx for embedding lookup."""
        sample = self.samples[idx]

        query_id = sample['query_id']
        pos_doc_id = sample['pos_doc_id']
        neg_doc_ids = sample['neg_doc_ids']

        # Get texts
        query_instruction = self.metadata.get('query_instruction', '')
        query_text = query_instruction + self.queries[query_id]

        pos_doc = self.corpus[pos_doc_id]
        pos_text = pos_doc.get('title', '') + ' ' + pos_doc.get('text', '')
        pos_text = pos_text.strip()

        neg_texts = []
        for neg_doc_id in neg_doc_ids:
            if neg_doc_id in self.corpus:
                neg_doc = self.corpus[neg_doc_id]
                neg_text = neg_doc.get('title', '') + ' ' + neg_doc.get('text', '')
                neg_texts.append(neg_text.strip())

        return {
            'query': query_text,
            'positive': pos_text,
            'negatives': neg_texts,
            '_query_idx': self.query_id_to_idx[query_id],
            '_pos_idx': self.corpus_id_to_idx[pos_doc_id],
            '_neg_indices': [self.corpus_id_to_idx[nid] for nid in neg_doc_ids if nid in self.corpus_id_to_idx]
        }

    def get_query_embedding(self, idx):
        """Get query embedding by sample index."""
        sample = self.samples[idx]
        query_idx = self.query_id_to_idx[sample['query_id']]
        return self.query_embeddings[query_idx]

    def get_positive_embedding(self, idx):
        """Get positive document embedding by sample index."""
        sample = self.samples[idx]
        pos_idx = self.corpus_id_to_idx[sample['pos_doc_id']]
        return self.corpus_embeddings[pos_idx]

    def get_negative_embeddings(self, idx):
        """Get negative document embeddings by sample index."""
        sample = self.samples[idx]
        neg_indices = [self.corpus_id_to_idx[nid] for nid in sample['neg_doc_ids'] if nid in self.corpus_id_to_idx]
        return self.corpus_embeddings[neg_indices]
