"""
Dataset utility functions for training.
"""
import random
import torch
import logging
from datasets import concatenate_datasets

# Get logger
logger = logging.getLogger(__name__)

def _get_size_mb(ds):
    """Roughly estimate dataset size in MB.

    1. Use the `dataset_size` provided in the dataset `info` dict when available.
    2. If the original `text` column is still present, sum the UTF-8 byte lengths of
       the examples.
    3. Otherwise (e.g. after tokenisation where only `input_ids` remain) estimate by
       counting tokens: `len(input_ids) * 2` bytes (2-byte per token rough guess).
    """
    size_bytes = None

    # 1) Try metadata if it exists
    if hasattr(ds, "info"):
        size_bytes = getattr(ds.info, "dataset_size", None) or getattr(ds.info, "size_in_bytes", None)

    # 2) Fallback: compute from columns
    if size_bytes is None:
        try:
            if "text" in ds.column_names:
                # Sum UTF-8 bytes of text
                size_bytes = sum(len(t.encode("utf-8")) for t in ds["text"])
            elif "input_ids" in ds.column_names:
                # Rough estimate: 2 bytes per token (uint16/INT)
                size_bytes = sum(len(ids) for ids in ds["input_ids"]) * 2
            else:
                size_bytes = 0
        except Exception:
            size_bytes = 0

    return (size_bytes or 0) / (1024 * 1024)


def _print_split_stats(name, train_ds, val_ds):
    """Print dataset split statistics."""
    print(f"{name} - Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"{name} sizes -> Train: {_get_size_mb(train_ds):.1f} MB , Val: {_get_size_mb(val_ds):.1f} MB , Total: {_get_size_mb(train_ds)+_get_size_mb(val_ds):.1f} MB")


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def combine_datasets(datasets, split='train'):
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of dataset objects
        split: Which split to combine ('train' or 'validation')
    
    Returns:
        Combined dataset
    """
    valid_datasets = []
    for ds in datasets:
        if ds is not None and len(ds) > 0:
            valid_datasets.append(ds)
    
    if not valid_datasets:
        logger.warning(f"No valid datasets found for {split} split")
        return None
    
    if len(valid_datasets) == 1:
        return valid_datasets[0]
    
    combined = concatenate_datasets(valid_datasets)
    logger.info(f"Combined {len(valid_datasets)} datasets into {len(combined)} samples for {split}")
    return combined
