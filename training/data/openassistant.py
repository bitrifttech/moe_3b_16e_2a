"""
OpenAssistant dataset processing and loading.
"""
import logging
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets
from ..utils.dataset_utils import _print_split_stats

# Get logger
logger = logging.getLogger(__name__)

def build_openassistant_conversations(ds):
    """Build proper conversation threads from OpenAssistant dataset."""
    # Group messages by conversation
    conversations = {}
    
    for message in ds:
        if message.get("lang") != "en":
            continue
            
        conv_id = message.get("conversation_id")
        if not conv_id:
            continue
            
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(message)
    
    # Build threaded conversations
    formatted_conversations = []
    
    for conv_id, messages in conversations.items():
        # Build message thread
        thread = build_message_thread(messages)
        if len(thread) >= 2:  # Need at least human + assistant
            formatted_text = format_conversation_thread(thread)
            if formatted_text:
                formatted_conversations.append({"text": formatted_text})
    
    return Dataset.from_list(formatted_conversations)


def build_message_thread(messages):
    """Build a single conversation thread from messages."""
    # Create a mapping of message_id to message
    msg_map = {msg["message_id"]: msg for msg in messages}
    
    # Find the root message (no parent)
    root_messages = [msg for msg in messages if not msg.get("parent_id")]
    if not root_messages:
        return []
    
    # Build thread starting from root
    thread = []
    current_msg = root_messages[0]  # Take first root
    
    while current_msg:
        thread.append(current_msg)
        
        # Find children of current message
        children = [msg for msg in messages if msg.get("parent_id") == current_msg["message_id"]]
        
        if children:
            # Take the highest ranked child (best response)
            current_msg = max(children, key=lambda x: x.get("rank", 0))
        else:
            current_msg = None
    
    return thread


def format_conversation_thread(thread):
    """Format conversation thread as instruction-response pairs."""
    formatted_parts = []
    
    for message in thread:
        role = message.get("role", "")
        text = message.get("text", "").strip()
        
        if not text:
            continue
        
        if role == "prompter":  # Human/user message
            formatted_parts.append(f"Human: {text}")
        elif role == "assistant":  # Assistant response
            formatted_parts.append(f"Assistant: {text}")
    
    if len(formatted_parts) >= 2:  # Need at least one exchange
        return "\n\n".join(formatted_parts)
    
    return ""


def load_openassistant_dataset(tok, max_len=256):
    """Load and preprocess the OpenAssistant OASST1 dataset."""
    print("Loading OpenAssistant OASST1 dataset from HuggingFace...")
    ds = load_dataset("OpenAssistant/oasst1")
    
    print("Processing dataset...")
    # If the dataset has multiple splits, concatenate them
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
    
    print("Building conversation threads...")
    # Build proper conversation threads instead of just extracting assistant responses
    conversations = build_openassistant_conversations(ds)
    
    print("Splitting into train/validation...")
    # Split into train and validation
    split = conversations.train_test_split(test_size=0.05, seed=42)
    
    def tok_fn(batch):
        # Tokenize without padding; dynamic padding will be added by the data collator
        return tok(batch["text"], truncation=True, max_length=max_len)
    
    print("Tokenizing training set...")
    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=split["train"].column_names)
    
    print("Tokenizing validation set...")
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=split["test"].column_names)
    
    # Remove examples that became empty after tokenization
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 1)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) > 1)
    
    _print_split_stats("OpenAssistant", train_ds, val_ds)
    
    return train_ds, val_ds
