"""
OpenOrca Dataset Loader

Loads and processes OpenOrca dataset for conversational training.
High-quality GPT-4 and GPT-3.5 completions based on FLAN Collection.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class OpenOrcaLoader(BaseDatasetLoader):
    """Loader for OpenOrca dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="OpenOrca",
                max_samples=100000,  # Large dataset, so we sample more
                max_length=1536,  # Increased from 512 for longer conversations
                min_length=50
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load OpenOrca dataset from HuggingFace."""
        try:
            self.logger.info("Loading OpenOrca dataset...")
            dataset = hf_load_dataset("Open-Orca/OpenOrca")
            
            if "train" in dataset:
                return dataset["train"]
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                self.logger.info(f"Using split '{split_name}' from OpenOrca")
                return dataset[split_name]
                
        except Exception as e:
            self.logger.error(f"Failed to load OpenOrca dataset: {e}")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """Preprocess OpenOrca raw data into standardized format."""
        processed_data = []
        
        for example in raw_data:
            processed_example = self.process_example(example)
            if processed_example is not None:
                processed_data.append(processed_example)
        
        return processed_data
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single OpenOrca example into conversation format."""
        try:
            # Check if already processed (has 'text' field)
            if "text" in example:
                text = example["text"].strip()
                
                # Quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                # Basic quality filter for preprocessed text
                lower_text = text.lower()
                bad_patterns = [
                    "i cannot",
                    "i can't help",
                    "i'm not able to", 
                    "i don't have access",
                    "as an ai language model",
                    "i'm just a computer program"
                ]
                
                # Count bad patterns in the text
                bad_count = sum(1 for pattern in bad_patterns if pattern in lower_text)
                if bad_count > 2:  # Allow some but not if they dominate
                    return None
                
                return {"text": text}
            
            # Original logic for raw format (fallback)
            elif "question" in example and "response" in example:
                # OpenOrca format typically has 'system_prompt', 'question', 'response'
                system_prompt = example.get("system_prompt", "").strip()
                question = example.get("question", "").strip()
                response = example.get("response", "").strip()
                
                if not question or not response:
                    return None
                
                # Build conversation
                conversation_parts = []
                
                # Add system context if available and meaningful
                if system_prompt and len(system_prompt) < 200:  # Keep system prompts reasonable
                    if not system_prompt.lower().startswith("you are a helpful assistant"):
                        conversation_parts.append(f"System: {system_prompt}")
                
                # Add the main conversation
                conversation_parts.append(f"Human: {question}")
                conversation_parts.append(f"Assistant: {response}")
                
                # Join conversation
                text = "\n\n".join(conversation_parts)
                
                # Quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                # Filter out low-quality responses
                lower_response = response.lower()
                bad_patterns = [
                    "i cannot",
                    "i can't help",
                    "i'm not able to",
                    "i don't have access",
                    "as an ai language model",
                    "i'm just a computer program"
                ]
                
                # Allow some of these patterns but not if they dominate the response
                bad_count = sum(1 for pattern in bad_patterns if pattern in lower_response)
                if bad_count > 1 or (bad_count == 1 and len(response) < 100):
                    return None
                
                # Prefer GPT-4 responses if we can identify them
                model_name = example.get("id", "").lower()
                if "gpt-4" in model_name:
                    # Mark high-quality responses (we could use this for sampling later)
                    pass
                
                return {"text": text}
            
            else:
                # Neither format recognized
                return None
            
        except Exception as e:
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the OpenOrca dataset."""
        return {
            "name": "OpenOrca",
            "description": "GPT-4 and GPT-3.5 completions based on FLAN Collection data",
            "source": "Open-Orca/OpenOrca",
            "total_size": "~4.2M examples (1M GPT-4, 3.2M GPT-3.5)",
            "quality": "Very High - GPT-4 quality responses",
            "use_case": "Strong reasoning, instruction-following, problem-solving"
        }

def create_openorca_loader(max_samples: int = 100000, max_length: int = 512) -> OpenOrcaLoader:
    """Factory function to create OpenOrca loader with custom config."""
    config = DatasetConfig(
        name="OpenOrca",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return OpenOrcaLoader(config)
