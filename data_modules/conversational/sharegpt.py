"""
ShareGPT Dataset Loader

Loads and processes ShareGPT dataset for conversational training.
Real ChatGPT conversations shared by users.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class ShareGPTLoader(BaseDatasetLoader):
    """Loader for ShareGPT dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="ShareGPT",
                max_samples=20000,
                max_length=500,
                min_length=30
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load ShareGPT dataset from HuggingFace."""
        # List of working ShareGPT datasets to try in order
        datasets_to_try = [
            "icybee/share_gpt_90k_v1",  # 90k conversations, well-maintained
            "Dans-DiscountModels/ConversationChronicles-sharegpt",  # 785k conversations
            "HuggingFaceH4/Bespoke-Stratos-17k",  # 17k high-quality conversations
            "arcee-ai/EvolKit-20k"  # 20k evolved conversations
        ]
        
        for dataset_name in datasets_to_try:
            try:
                self.logger.info(f"Attempting to load ShareGPT dataset: {dataset_name}")
                dataset = hf_load_dataset(dataset_name)
                
                # Handle different dataset structures
                if "train" in dataset:
                    return dataset["train"]
                elif len(dataset) > 0:
                    # Take the first available split
                    split_name = list(dataset.keys())[0]
                    self.logger.info(f"Using split '{split_name}' from {dataset_name}")
                    return dataset[split_name]
                else:
                    raise ValueError(f"No usable splits found in {dataset_name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        # If all datasets fail, return empty list
        self.logger.error("All ShareGPT dataset sources failed")
        return []
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process ShareGPT data into conversation format.
        
        Handles multiple ShareGPT formats and extracts assistant responses.
        """
        processed_data = []
        
        for item in raw_data:
            # Handle different ShareGPT formats
            conversations = None
            
            # Format 1: Standard ShareGPT with "conversations" field
            if "conversations" in item:
                conversations = item["conversations"]
            # Format 2: Direct messages field (some datasets use this)
            elif "messages" in item:
                conversations = item["messages"]
            # Format 3: Some datasets have conversation data in other fields
            elif "conversation" in item:
                conversations = item["conversation"]
            # Format 4: Some use "turns" or "dialogue"
            elif "turns" in item or "dialogue" in item:
                conversations = item.get("turns", item.get("dialogue", []))
            
            if not conversations:
                continue
            
            # Extract assistant/gpt responses from conversations
            for conv in conversations:
                # Handle different role naming conventions
                role = conv.get("from", conv.get("role", "")).lower()
                content = conv.get("value", conv.get("content", "")).strip()
                
                # Look for assistant/gpt responses
                if role in ["gpt", "assistant", "bot", "chatgpt"] and content:
                    processed_data.append({
                        "text": content,
                        "metadata": {
                            "source": "sharegpt",
                            "conversation_id": item.get("id", item.get("conversation_id", "")),
                            "type": "assistant_response",
                            "role": role
                        }
                    })
        
        return processed_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply ShareGPT-specific filters."""
        text = item["text"]
        
        # Filter out very short responses (reduced from 10 to 5 words)
        if len(text.split()) < 5:
            return False
        
        # Filter out responses that are mostly code (made more lenient)
        if text.count("```") >= 2:  # Has code blocks
            code_lines = 0
            total_lines = len(text.split('\n'))
            in_code_block = False
            
            for line in text.split('\n'):
                if "```" in line:
                    in_code_block = not in_code_block
                elif in_code_block:
                    code_lines += 1
            
            # Allow up to 70% code content instead of 50%
            if code_lines / max(total_lines, 1) > 0.7:
                return False
        
        # Filter out responses with too many special characters (made more lenient)
        # Allow up to 15% special characters instead of 10%
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-()[]{}":;\n\t') / len(text)
        if special_char_ratio > 0.15:
            return False
        
        return True

def create_sharegpt_loader(max_samples: int = 20000, max_length: int = 500) -> ShareGPTLoader:
    """Factory function to create ShareGPT loader with custom config."""
    config = DatasetConfig(
        name="ShareGPT",
        max_samples=max_samples,
        max_length=max_length,
        min_length=30
    )
    return ShareGPTLoader(config)
