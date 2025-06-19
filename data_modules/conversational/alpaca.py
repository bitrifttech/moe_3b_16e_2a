"""
Alpaca Dataset Loader

Loads and processes the Stanford Alpaca dataset for instruction-following training.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class AlpacaLoader(BaseDatasetLoader):
    """Loader for Stanford Alpaca dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="Alpaca",
                max_samples=15000,
                max_length=300,
                min_length=20
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load Alpaca dataset from HuggingFace."""
        from datasets import load_dataset
        try:
            # Try the main Alpaca dataset
            dataset = load_dataset("tatsu-lab/alpaca")
            return dataset["train"]
        except Exception as e:
            self.logger.warning(f"Failed to load main Alpaca dataset: {e}")
            # Try alternative Alpaca sources
            try:
                dataset = hf_load_dataset("yahma/alpaca-cleaned")
                return dataset["train"]
            except Exception as e2:
                self.logger.error(f"All Alpaca sources failed: {e2}")
                raise e
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process Alpaca data into instruction-response format.
        
        Combines instruction, input, and output into conversational format.
        """
        processed_data = []
        
        for item in raw_data:
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            
            if not instruction or not output:
                continue
            
            # Format as instruction-following conversation
            if input_text:
                # Has additional input context
                conversation = f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {output}"
            else:
                # Simple instruction-response
                conversation = f"Instruction: {instruction}\n\nResponse: {output}"
            
            processed_data.append({
                "text": conversation,
                "metadata": {
                    "source": "alpaca",
                    "type": "instruction_following",
                    "has_input": bool(input_text)
                }
            })
        
        return processed_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply Alpaca-specific filters."""
        text = item["text"]
        
        # Filter out very short or very long responses
        word_count = len(text.split())
        if word_count < 15 or word_count > 400:
            return False
        
        # Filter out responses that are mostly lists or bullet points
        line_count = len(text.split('\n'))
        if line_count > 10:  # Too many lines, likely a long list
            bullet_lines = sum(1 for line in text.split('\n') if line.strip().startswith(('-', '*', '•', '1.', '2.')))
            if bullet_lines / line_count > 0.5:  # More than 50% bullet points
                return False
        
        # Keep responses that look like natural instructions and responses
        return True

def create_alpaca_loader(max_samples: int = 15000, max_length: int = 300) -> AlpacaLoader:
    """Factory function to create Alpaca loader with custom config."""
    config = DatasetConfig(
        name="Alpaca",
        max_samples=max_samples,
        max_length=max_length,
        min_length=20
    )
    return AlpacaLoader(config)
