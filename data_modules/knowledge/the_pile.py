"""
The Pile Dataset Loader

Loads and processes The Pile dataset for diverse text training.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class ThePileLoader(BaseDatasetLoader):
    """Loader for The Pile dataset."""
    
    def __init__(self, config: DatasetConfig = None, subset: str = None):
        if config is None:
            config = DatasetConfig(
                name="The-Pile",
                max_samples=30000,
                max_length=800,
                min_length=50
            )
        super().__init__(config)
        self.subset = subset  # Can specify a specific pile subset
    
    def load_raw_data(self) -> Any:
        """Load The Pile dataset from HuggingFace."""
        try:
            if self.subset:
                dataset = hf_load_dataset("the_pile", split=f"train[:{self.config.max_samples}]")
            else:
                # Load a smaller, more manageable subset
                dataset = hf_load_dataset("the_pile", "all", streaming=True)
            return dataset["train"]
        except Exception as e:
            self.logger.error(f"Failed to load The Pile dataset: {e}")
            # Fallback to a smaller pile-like dataset
            try:
                dataset = load_dataset("EleutherAI/pile-uncopyrighted")
                return dataset["train"]
            except Exception as e2:
                self.logger.error(f"Pile fallback failed: {e2}")
                raise e
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process The Pile data.
        
        Extracts text from various pile sources.
        """
        processed_data = []
        
        # Handle streaming dataset
        count = 0
        max_samples = self.config.max_samples or 30000
        
        for item in raw_data:
            if count >= max_samples:
                break
                
            text = item.get("text", "").strip()
            meta = item.get("meta", {})
            
            if not text:
                continue
            
            # Get pile source information
            pile_set_name = meta.get("pile_set_name", "unknown")
            
            processed_data.append({
                "text": text,
                "metadata": {
                    "source": "the_pile",
                    "pile_set": pile_set_name,
                    "type": "diverse_text"
                }
            })
            
            count += 1
        
        return processed_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply The Pile-specific filters."""
        text = item["text"]
        pile_set = item["metadata"].get("pile_set", "")
        
        # Filter out certain pile sets that might not be good for chat training
        excluded_sets = {
            "DM Mathematics",  # Too specialized
            "ArXiv",          # Academic papers might be too technical
            "USPTO Backgrounds",  # Legal text
            "FreeLaw"         # Legal text
        }
        
        if pile_set in excluded_sets:
            return False
        
        # Filter out very short or very long texts
        word_count = len(text.split())
        if word_count < 30 or word_count > 1000:
            return False
        
        # Filter out texts with too much code (basic heuristic)
        code_indicators = ["def ", "class ", "import ", "function", "var ", "const ", "let "]
        code_count = sum(text.count(indicator) for indicator in code_indicators)
        if code_count > 5:  # Likely a code file
            return False
        
        # Filter out texts with too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-()[]{}":;\n') / len(text)
        if special_char_ratio > 0.2:
            return False
        
        return True

def create_the_pile_loader(
    subset: str = None,
    max_samples: int = 30000, 
    max_length: int = 800
) -> ThePileLoader:
    """Factory function to create The Pile loader with custom config."""
    config = DatasetConfig(
        name=f"The-Pile{'-' + subset if subset else ''}",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return ThePileLoader(config, subset)
