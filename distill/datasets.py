"""Dataset classes for distillation."""

import logging
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class Phase1Dataset(Dataset):
    """Dataset for Phase 1: General distillation with diverse data

    Combines MS MARCO, NQ, and HotpotQA for general semantic matching
    """
    def __init__(self, max_samples_per_dataset=100000):
        self.samples = []

        logger.info("Loading Phase 1 datasets...")

        # Load MS MARCO passages
        try:
            logger.info("Loading MS MARCO...")
            # Use streaming mode to avoid feature type compatibility issues
            msmarco = load_dataset('ms_marco', 'v1.1', split='train', streaming=True)
            for i, item in enumerate(msmarco):
                if i >= max_samples_per_dataset:
                    break
                if item['passages']['is_selected'][0]:  # Has relevant passage
                    query = item['query']
                    passage = item['passages']['passage_text'][0]
                    self.samples.append((query, passage, 'msmarco'))
            logger.info(f"Loaded {len([s for s in self.samples if s[2]=='msmarco'])} MS MARCO samples")
        except Exception as e:
            logger.warning(f"Could not load MS MARCO: {e}")

        logger.info(f"Total Phase 1 samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, passage, source = self.samples[idx]
        return {'query': query, 'passage': passage, 'source': source}


class Phase2Dataset(Dataset):
    """Dataset for Phase 2: MS MARCO with hard negatives

    Uses provided negatives from MS MARCO + in-batch negatives during training
    """
    def __init__(self, split='train', max_samples=500000):
        self.samples = []

        logger.info(f"Loading MS MARCO for Phase 2 ({split} split)...")

        try:
            # Use streaming mode to avoid feature type compatibility issues
            dataset = load_dataset('ms_marco', 'v1.1', split=split, streaming=True)

            for i, item in enumerate(tqdm(dataset, desc="Processing samples", total=max_samples)):
                if i >= max_samples:
                    break

                query = item['query']
                passages = item['passages']

                # Find positive passage
                positive_idx = None
                for idx, is_selected in enumerate(passages['is_selected']):
                    if is_selected:
                        positive_idx = idx
                        break

                if positive_idx is None:
                    continue

                positive_passage = passages['passage_text'][positive_idx]

                # Collect negative passages (non-selected ones)
                negative_passages = [
                    passages['passage_text'][idx]
                    for idx, is_selected in enumerate(passages['is_selected'])
                    if not is_selected
                ]

                # Take up to 7 negatives
                negative_passages = negative_passages[:7]

                if len(negative_passages) > 0:
                    self.samples.append({
                        'query': query,
                        'positive': positive_passage,
                        'negatives': negative_passages
                    })

            logger.info(f"Loaded {len(self.samples)} Phase 2 samples with negatives")
        except Exception as e:
            logger.warning(f"Could not load MS MARCO for Phase 2: {e}")

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
