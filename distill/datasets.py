"""Dataset classes for distillation."""

import logging
import os
import random
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


class Phase1Dataset(Dataset):
    """Dataset for Phase 1: General distillation with diverse data

    Supports loading multiple datasets from BEIR benchmark using the beir library.
    Datasets are downloaded and cached locally for efficient loading.

    For Phase 1 (pure distillation), queries and documents are randomly sampled
    separately with a 1:19 ratio (5% queries, 95% documents). Queries get an
    instruction prefix while documents don't. No pairing is needed.

    Returns a simple list of strings (not dictionaries) for uniform processing
    during training without query/document discrimination.

    Supported datasets: msmarco, nfcorpus, trec-covid, nq, hotpotqa, fiqa,
                       arguana, scidocs, scifact, touche-2020, quora,
                       dbpedia-entity, fever, climate-fever, signal1m

    Args:
        datasets_config: List of dataset configurations, each with 'name' and 'max_samples'
                        Example: [{'name': 'msmarco', 'max_samples': 100000},
                                 {'name': 'nfcorpus', 'max_samples': 10000}]
        max_samples_per_dataset: (deprecated) Fallback for backward compatibility
        data_dir: Directory to store downloaded BEIR datasets (default: './beir_datasets')

    Returns:
        String: A text sample (either query with prefix or document without prefix)
    """
    def __init__(self, datasets_config=None, max_samples_per_dataset=100000, data_dir='./beir_datasets'):
        self.samples = []
        self.data_dir = data_dir

        # Backward compatibility: if no config provided, use default MS MARCO
        if datasets_config is None:
            datasets_config = [{'name': 'msmarco', 'max_samples': max_samples_per_dataset}]

        logger.info("Loading Phase 1 datasets...")

        for dataset_cfg in datasets_config:
            dataset_name = dataset_cfg['name'].lower()
            max_samples = dataset_cfg.get('max_samples', max_samples_per_dataset)

            try:
                # List of datasets available in BEIR
                beir_datasets = ['msmarco', 'nfcorpus', 'trec-covid', 'nq', 'hotpotqa',
                                 'fiqa', 'arguana', 'scidocs', 'scifact',
                                 'touche-2020', 'quora', 'dbpedia-entity',
                                 'fever', 'climate-fever', 'signal1m']

                if dataset_name in beir_datasets:
                    self._load_beir_dataset(dataset_name, max_samples)
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}, skipping...")
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")

        logger.info(f"Total Phase 1 samples: {len(self.samples)}")

    def _load_beir_dataset(self, dataset_name, max_samples):
        """Load BEIR dataset using beir library.

        For Phase 1 (distillation), randomly samples queries and documents separately
        with a 1:19 ratio. Queries get instruction prefix, documents don't.
        """
        logger.info(f"Loading BEIR dataset '{dataset_name}' (max {max_samples} samples)...")

        # Download and unzip dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, self.data_dir)

        # Load corpus and queries (no need for qrels in Phase 1)
        corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="train")

        # Sample with 1:19 ratio (1 query : 19 docs)
        # Total samples = max_samples, so queries = max_samples / 20, docs = max_samples * 19 / 20
        num_queries = max_samples // 20
        num_docs = max_samples - num_queries

        # Extract and add queries with instruction prefix (no shuffling needed)
        query_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        count_queries = 0
        for query_id, query_text in queries.items():
            if count_queries >= num_queries:
                break
            if query_text.strip():
                self.samples.append(query_instruction + query_text)
                count_queries += 1

        # Extract and add documents (no shuffling needed)
        count_docs = 0
        for doc_id, doc in corpus.items():
            if count_docs >= num_docs:
                break
            doc_text = doc.get('title', '') + ' ' + doc.get('text', '')
            doc_text = doc_text.strip()
            if doc_text:
                self.samples.append(doc_text)
                count_docs += 1

        random.shuffle(self.samples)
        logger.info(f"Loaded {count_queries} queries and {count_docs} docs from {dataset_name} (total: {count_queries + count_docs} samples)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Phase2Dataset(Dataset):
    """Dataset for Phase 2: Task-specific training with hard negatives

    Supports loading multiple datasets from BEIR benchmark using the beir library.
    Datasets are downloaded and cached locally. Creates training samples with
    one positive and multiple hard negative documents per query.

    Supported datasets: msmarco, nfcorpus, trec-covid, nq, hotpotqa, fiqa,
                       arguana, scidocs, scifact, touche-2020, quora,
                       dbpedia-entity, fever, climate-fever, signal1m

    Args:
        datasets_config: List of dataset configurations, each with 'name' and 'max_samples'
                        Example: [{'name': 'msmarco', 'max_samples': 500000},
                                 {'name': 'nfcorpus', 'max_samples': 50000}]
        split: Dataset split to use (default: 'train')
        num_negatives: Number of hard negatives per query (default: 7)
        data_dir: Directory to store downloaded BEIR datasets (default: './beir_datasets')
        dataset_name: (deprecated) Fallback for backward compatibility
        max_samples: (deprecated) Fallback for backward compatibility
    """
    def __init__(self, datasets_config=None, split='train', num_negatives=7, data_dir='./beir_datasets', dataset_name=None, max_samples=None):
        self.samples = []
        self.num_negatives = num_negatives
        self.data_dir = data_dir
        self.split = split

        # Backward compatibility: if no config provided, use old API
        if datasets_config is None:
            if dataset_name is None:
                dataset_name = 'msmarco'
            if max_samples is None:
                max_samples = 500000
            datasets_config = [{'name': dataset_name, 'max_samples': max_samples}]

        logger.info("Loading Phase 2 datasets...")

        for dataset_cfg in datasets_config:
            dataset_name = dataset_cfg['name'].lower()
            max_samples = dataset_cfg.get('max_samples', 500000)

            try:
                # List of datasets available in BEIR
                beir_datasets = ['msmarco', 'nfcorpus', 'trec-covid', 'nq', 'hotpotqa',
                                 'fiqa', 'arguana', 'scidocs', 'scifact',
                                 'touche-2020', 'quora', 'dbpedia-entity',
                                 'fever', 'climate-fever', 'signal1m']

                if dataset_name in beir_datasets:
                    self._load_beir_dataset(dataset_name, max_samples)
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}, skipping...")
            except Exception as e:
                logger.warning(f"Could not load {dataset_name} for Phase 2: {e}")

        logger.info(f"Total Phase 2 samples: {len(self.samples)}")

    def _load_beir_dataset(self, dataset_name, max_samples):
        """Load BEIR dataset with hard negatives using beir library.

        Downloads the dataset if not cached and loads corpus, queries, and qrels.
        Creates samples with one positive and multiple hard negative documents.
        """
        logger.info(f"Loading BEIR dataset '{dataset_name}' for Phase 2 (max {max_samples} samples)...")

        # Download and unzip dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, self.data_dir)

        # Load corpus, queries, and qrels using GenericDataLoader
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=self.split)

        # Convert corpus keys to list once for efficient sampling
        all_corpus_ids = list(corpus.keys())

        # Create query-document pairs with hard negatives from qrels
        count = 0
        for query_id, doc_scores in tqdm(qrels.items(), desc=f"Processing {dataset_name} samples", total=max_samples):
            if count >= max_samples:
                break

            if query_id not in queries:
                continue

            query_text = queries[query_id]

            # Separate positive and negative documents based on relevance score
            positive_docs = [(doc_id, score) for doc_id, score in doc_scores.items() if score > 0]
            negative_docs = [(doc_id, score) for doc_id, score in doc_scores.items() if score == 0]

            if not positive_docs:
                continue

            # Use the document with highest relevance score as positive
            pos_doc_id, _ = max(positive_docs, key=lambda x: x[1])

            if pos_doc_id not in corpus:
                continue

            # Get positive passage text (combine title and text)
            pos_doc = corpus[pos_doc_id]
            positive_text = pos_doc.get('title', '') + ' ' + pos_doc.get('text', '')
            positive_text = positive_text.strip()

            if not positive_text:
                continue

            # Collect hard negatives
            negative_passages = []
            for neg_doc_id, _ in negative_docs:
                if len(negative_passages) >= self.num_negatives:
                    break

                if neg_doc_id in corpus:
                    neg_doc = corpus[neg_doc_id]
                    neg_text = neg_doc.get('title', '') + ' ' + neg_doc.get('text', '')
                    neg_text = neg_text.strip()
                    if neg_text:
                        negative_passages.append(neg_text)

            # If not enough explicit negatives, sample random documents
            if len(negative_passages) < self.num_negatives:
                # Get document IDs not in this query's relevance judgments
                judged_doc_ids = set(doc_scores.keys())
                num_needed = self.num_negatives - len(negative_passages)

                # Randomly sample needed negatives (with some buffer for empty docs)
                sample_size = min(num_needed * 3, len(all_corpus_ids))  # 3x buffer for empty/invalid docs
                sampled_ids = random.sample(all_corpus_ids, sample_size)

                for doc_id in sampled_ids:
                    if len(negative_passages) >= self.num_negatives:
                        break

                    # Skip judged documents
                    if doc_id in judged_doc_ids:
                        continue

                    neg_doc = corpus[doc_id]
                    neg_text = neg_doc.get('title', '') + ' ' + neg_doc.get('text', '')
                    neg_text = neg_text.strip()
                    if neg_text:
                        negative_passages.append(neg_text)

            # Only add sample if we have at least one negative
            if negative_passages:
                self.samples.append({
                    'query': query_text,
                    'positive': positive_text,
                    'negatives': negative_passages[:self.num_negatives]
                })
                count += 1

        logger.info(f"Loaded {count} {dataset_name} samples with hard negatives")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def phase2_collate_fn(batch):
    """Custom collate function to handle variable-length negatives

    Pads negatives to the same length within batch
    """
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives_list = [item['negatives'] for item in batch]

    # Find max number of negatives in this batch
    max_negatives = max(len(negs) for negs in negatives_list)

    # Pad negatives to same length (repeat last negative if needed)
    padded_negatives = []
    for negs in negatives_list:
        if len(negs) < max_negatives:
            # Pad by repeating the last negative
            padded = negs + [negs[-1]] * (max_negatives - len(negs))
        else:
            padded = negs
        padded_negatives.append(padded)

    return {
        'query': queries,
        'positive': positives,
        'negatives': padded_negatives
    }
