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
        try:
            # Try the main ShareGPT dataset
            dataset = hf_load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")
            return dataset["train"]
        except Exception as e:
            self.logger.warning(f"Failed to load ShareGPT dataset: {e}")
            # Fallback to a smaller, more reliable ShareGPT variant
            try:
                dataset = hf_load_dataset("philschmid/sharegpt-raw")
                return dataset["train"]
            except Exception as e2:
                self.logger.error(f"All ShareGPT sources failed: {e2}")
                raise e
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process ShareGPT data into conversation format.
        
        Extracts assistant responses from conversations.
        """
        processed_data = []
        
        for item in raw_data:
            conversations = item.get("conversations", [])
            if not conversations:
                continue
            
            # Extract assistant responses
            for conv in conversations:
                if conv.get("from") == "gpt" or conv.get("role") == "assistant":
                    text = conv.get("value", "").strip()
                    if not text:
                        continue
                    
                    processed_data.append({
                        "text": text,
                        "metadata": {
                            "source": "sharegpt",
                            "conversation_id": item.get("id", ""),
                            "type": "assistant_response"
                        }
                    })
        
        return processed_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply ShareGPT-specific filters."""
        text = item["text"]
        
        # Filter out very short responses
        if len(text.split()) < 10:
            return False
        
        # Filter out responses that are mostly code
        if text.count("```") >= 2:  # Has code blocks
            code_lines = 0
            total_lines = len(text.split('\n'))
            in_code_block = False
            
            for line in text.split('\n'):
                if "```" in line:
                    in_code_block = not in_code_block
                elif in_code_block:
                    code_lines += 1
            
            if code_lines / max(total_lines, 1) > 0.5:  # More than 50% code
                return False
        
        # Filter out responses with too many special characters (likely corrupted)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-()[]{}":;') / len(text)
        if special_char_ratio > 0.1:  # More than 10% special characters
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
