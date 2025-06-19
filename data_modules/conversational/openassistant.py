"""
OpenAssistant Dataset Loader

Loads and processes the OpenAssistant/oasst1 dataset for conversational training.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class OpenAssistantLoader(BaseDatasetLoader):
    """Loader for OpenAssistant/oasst1 dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="OpenAssistant",
                max_samples=25000,
                max_length=400,
                min_length=20
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load OpenAssistant dataset from HuggingFace."""
        try:
            dataset = hf_load_dataset("OpenAssistant/oasst1")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load OpenAssistant dataset: {e}")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process OpenAssistant data into conversation format.
        
        Extracts assistant responses from the conversation trees.
        """
        processed_data = []
        
        # Get the training split
        train_data = raw_data["train"]
        
        # Filter for English assistant responses
        assistant_messages = train_data.filter(
            lambda x: x["role"] == "assistant" and x["lang"] == "en"
        )
        
        for message in assistant_messages:
            text = message["text"].strip()
            if not text:
                continue
                
            # Create conversation context if parent exists
            conversation_text = text
            
            processed_data.append({
                "text": conversation_text,
                "metadata": {
                    "source": "openassistant",
                    "message_id": message.get("message_id", ""),
                    "quality": message.get("rank", 0)
                }
            })
        
        return processed_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply OpenAssistant-specific filters."""
        text = item["text"]
        
        # Filter out very short responses
        if len(text.split()) < 5:
            return False
            
        # Filter out responses that are mostly code (basic heuristic)
        code_indicators = ["```", "def ", "class ", "import ", "function("]
        code_ratio = sum(1 for indicator in code_indicators if indicator in text) / len(code_indicators)
        if code_ratio > 0.3:  # If more than 30% code indicators
            return False
            
        return True

def create_openassistant_loader(max_samples: int = 25000, max_length: int = 400) -> OpenAssistantLoader:
    """Factory function to create OpenAssistant loader with custom config."""
    config = DatasetConfig(
        name="OpenAssistant",
        max_samples=max_samples,
        max_length=max_length,
        min_length=20
    )
    return OpenAssistantLoader(config)
