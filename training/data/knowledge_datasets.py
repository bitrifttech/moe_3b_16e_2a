"""
Knowledge dataset loaders (Pile, Wikipedia).
"""
import logging
from datasets import load_dataset, concatenate_datasets
from ..utils.dataset_utils import _print_split_stats

# Get logger
logger = logging.getLogger(__name__)

def load_pile_dataset(tokenizer, max_len=256, max_samples=100000):
    """Load and preprocess a subset of The Pile dataset."""
    print("Loading The Pile dataset from HuggingFace...")
    
    # Load a subset of The Pile
    datasets = []
    subsets = ["enron_emails", "pubmed"]  # Using smaller subsets that fit in ~1GB
    
    for subset in subsets:
        try:
            print(f"Loading {subset} subset...")
            ds = load_dataset("EleutherAI/pile", subset, split=f"train[:{max_samples//len(subsets)}]")
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {subset}: {str(e)}")
    
    if not datasets:
        raise ValueError("Failed to load any Pile subsets")
        
    # Combine all subsets
    combined = concatenate_datasets(datasets)
    
    # Filter out very short or long examples
    combined = combined.filter(lambda x: 50 < len(x["text"].split()) < 1000)
    
    # Split into train and validation
    split = combined.train_test_split(test_size=0.05, seed=42)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding="max_length"
        )
    
    print("Tokenizing training set...")
    train_ds = split["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["train"].column_names,
        num_proc=4
    )
    
    print("Tokenizing validation set...")
    val_ds = split["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["test"].column_names,
        num_proc=4
     )
     
    _print_split_stats("The Pile", train_ds, val_ds)
    return train_ds, val_ds


def load_wikipedia_dataset(tokenizer, max_len=256, language="20220301.en", percent=1):
    """Load and preprocess a slice of the Wikipedia dump.
    `percent` specifies the percentage slice of the full dump to keep (1 ≈ ~250 MB for English).
    """
    slice_str = f"train[:{percent}%]" if percent < 100 else "train"
    print(f"Loading Wikipedia ({language}) dataset slice {slice_str} …")

    ds = load_dataset("wikipedia", language, split=slice_str)

    # Filter very short articles
    ds = ds.filter(lambda x: len(x["text"].split()) > 50)

    # Train/val split
    split = ds.train_test_split(test_size=0.05, seed=42)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    print("Tokenizing Wikipedia training set…")
    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=split["train"].column_names,
                                   num_proc=4)

    print("Tokenizing Wikipedia validation set…")
    val_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=split["test"].column_names,
                                  num_proc=4)
 
    _print_split_stats("Wikipedia", train_ds, val_ds)
    return train_ds, val_ds
