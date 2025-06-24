"""
UltraChat Dataset Loader

Loads and processes UltraChat 200k dataset for conversational training.
High-quality filtered conversations used to train Zephyr-7B-β.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class UltraChatLoader(BaseDatasetLoader):
    """Loader for UltraChat 200k dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="UltraChat",
                max_samples=50000,
                max_length=1536,  # ~384 tokens (4 chars per token average)
                min_length=50
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load UltraChat 200k dataset from HuggingFace."""
        try:
            self.logger.info("Loading UltraChat 200k dataset...")
            dataset = hf_load_dataset("HuggingFaceH4/ultrachat_200k")
            
            # Use the SFT (supervised fine-tuning) split
            if "train_sft" in dataset:
                return dataset["train_sft"]
            elif "sft" in dataset:
                return dataset["sft"]
            elif "train" in dataset:
                return dataset["train"]
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                self.logger.info(f"Using split '{split_name}' from UltraChat")
                return dataset[split_name]
                
        except Exception as e:
            self.logger.error(f"Failed to load UltraChat dataset: {e}")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """Preprocess UltraChat raw data into standardized format."""
        processed_data = []
        
        for example in raw_data:
            processed_example = self.process_example(example)
            if processed_example is not None:
                processed_data.append(processed_example)
        
        return processed_data
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single UltraChat example into conversation format."""
        try:
            # Check if already processed (has 'text' field)
            if "text" in example:
                text = example["text"].strip()
                
                # Quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                # Check for robotic responses (already filtered in UltraChat 200k, but double-check)
                lower_text = text.lower()
                robotic_phrases = [
                    "i don't have emotions",
                    "i do not have emotions", 
                    "i don't have opinions",
                    "i do not have opinions",
                    "as an ai",
                    "i'm just an ai"
                ]
                
                if any(phrase in lower_text for phrase in robotic_phrases):
                    return None
                
                return {"text": text}
            
            # Original logic for raw messages format (fallback)
            elif "messages" in example:
                messages = example["messages"]
                
                if not messages or len(messages) < 2:
                    return None
                
                # Extract conversation
                conversation_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "").strip()
                    
                    if not content:
                        continue
                        
                    if role == "user":
                        conversation_parts.append(f"Human: {content}")
                    elif role == "assistant":
                        conversation_parts.append(f"Assistant: {content}")
                
                if len(conversation_parts) < 2:
                    return None
                
                # Join conversation with newlines
                text = "\n\n".join(conversation_parts)
                
                # Quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                # Check for robotic responses
                lower_text = text.lower()
                robotic_phrases = [
                    "i don't have emotions",
                    "i do not have emotions", 
                    "i don't have opinions",
                    "i do not have opinions",
                    "as an ai",
                    "i'm just an ai"
                ]
                
                if any(phrase in lower_text for phrase in robotic_phrases):
                    return None
                
                return {"text": text}
            
            else:
                # Neither text nor messages format
                return None
                
        except Exception as e:
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the UltraChat dataset."""
        return {
            "name": "UltraChat 200k",
            "description": "High-quality filtered conversations used to train Zephyr-7B-β",
            "source": "HuggingFaceH4/ultrachat_200k",
            "total_size": "~200k conversations",
            "quality": "Very High - filtered and truecased",
            "use_case": "Natural conversation flow, reduced robotic responses"
        }

def create_ultrachat_loader(max_samples: int = 50000, max_length: int = 1536) -> UltraChatLoader:
    """Factory function to create UltraChat loader with custom config."""
    config = DatasetConfig(
        name="UltraChat",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return UltraChatLoader(config)
