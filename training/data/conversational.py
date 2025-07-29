"""
Conversational dataset processing and loading.
"""
import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict
from ..utils.dataset_utils import _print_split_stats

# Get logger
logger = logging.getLogger(__name__)

def load_conversational_dataset(tokenizer, max_len=256, max_samples=50000):
    """
    Load and preprocess the Anthropic HH-RLHF dataset for conversational training.
    This dataset contains high-quality human-AI conversations that will improve
    the model's conversational abilities.
    """
    print("Loading Anthropic HH-RLHF conversational dataset...")
    
    # Load the helpful subset of the HH-RLHF dataset
    try:
        print(f"Loading helpful-base subset...")
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        ds = ds["train"]  # Use training split
        
        # Take a subset if specified
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            
        print(f"Loaded {len(ds)} conversational examples")
        
        def format_conversation(example):
            """Format the conversation into proper instruction-response format"""
            chosen = example["chosen"]
            # Clean up the conversation format
            text = chosen.strip()
            
            # Ensure proper Human/Assistant formatting
            if "\n\nHuman:" in text and "\n\nAssistant:" in text:
                # Already properly formatted
                formatted_text = text
            else:
                # Treat as assistant response if not properly formatted
                formatted_text = f"Assistant: {text}"
            
            # Ensure it ends with an EOS token
            if not formatted_text.endswith(tokenizer.eos_token):
                formatted_text += tokenizer.eos_token
            return {"text": formatted_text}
        
        # Format conversations
        ds = ds.map(format_conversation, remove_columns=ds.column_names)
        
        # Filter out very short or very long conversations
        ds = ds.filter(lambda x: 50 <= len(x["text"].split()) <= 400)
        
        print(f"After filtering: {len(ds)} conversational examples")
        
        # Split into train/validation
        split = ds.train_test_split(test_size=0.05, seed=42)
        
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_len,
                return_tensors=None
            )
        
        # Tokenize
        train_ds = split["train"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["train"].column_names,
            num_proc=4
        )
        val_ds = split["test"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["test"].column_names,
            num_proc=4
        )
        
        print(f"Conversational dataset - Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds
        
    except Exception as e:
        print(f"Failed to load Anthropic HH-RLHF dataset: {e}")
        return _load_fallback_conversational_dataset(tokenizer, max_len, max_samples)


def _load_fallback_conversational_dataset(tokenizer, max_len=256, max_samples=50000):
    """Fallback conversational dataset loader using OpenAssistant."""
    print("Loading fallback conversational dataset (OpenAssistant)...")
    
    try:
        # Use a subset of the original OpenAssistant but format differently for conversations
        ds = load_dataset("OpenAssistant/oasst1")
        
        if isinstance(ds, DatasetDict):
            ds = concatenate_datasets([ds[k] for k in ds.keys()])
        
        # Filter for English conversations (both human and assistant)
        ds = ds.filter(lambda x: x["lang"] == "en" and len(x["text"].split()) > 20)
        
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        
        def format_oasst_conversation(example):
            text = example["text"].strip()
            
            # Ensure proper conversation formatting
            if example.get("role") == "assistant":
                # Format as assistant response with context if available
                formatted_text = f"Assistant: {text}"
            elif example.get("role") == "prompter":
                # Format as human prompt
                formatted_text = f"Human: {text}"
            else:
                # Default to assistant response
                formatted_text = f"Assistant: {text}"
            
            if not formatted_text.endswith(tokenizer.eos_token):
                formatted_text += tokenizer.eos_token
            return {"text": formatted_text}
        
        ds = ds.map(format_oasst_conversation, remove_columns=ds.column_names)
        split = ds.train_test_split(test_size=0.05, seed=42)
        
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_len,
                return_tensors=None
            )
        
        train_ds = split["train"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["train"].column_names,
            num_proc=4
        )
        val_ds = split["test"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["test"].column_names,
            num_proc=4
        )
        
        print(f"Fallback conversational dataset - Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds
        
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        print("No conversational dataset loaded")
        return None, None
