"""
Anthropic HH-RLHF Dataset Loader

Loads and processes the Anthropic HH-RLHF dataset for conversational training.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class AnthropicHHLoader(BaseDatasetLoader):
    """Loader for Anthropic HH-RLHF dataset."""
    
    def __init__(self, config: DatasetConfig = None, subset: str = "helpful-base"):
        if config is None:
            config = DatasetConfig(
                name="Anthropic-HH-RLHF",
                max_samples=30000,
                max_length=400,
                min_length=50
            )
        super().__init__(config)
        self.subset = subset  # helpful-base, helpful-rejection-sampled, harmless-base
    
    def load_raw_data(self) -> Any:
        """Load Anthropic HH-RLHF dataset from HuggingFace."""
        try:
            self.logger.info(f"Loading {self.subset} subset...")
            dataset = hf_load_dataset("Anthropic/hh-rlhf", data_dir=self.subset)
            return dataset["train"]  # Use training split
        except Exception as e:
            self.logger.error(f"Failed to load Anthropic HH-RLHF dataset: {e}")
            # Fallback to OpenAssistant if Anthropic fails
            self.logger.info("Falling back to OpenAssistant dataset...")
            try:
                dataset = hf_load_dataset("OpenAssistant/oasst1")
                return dataset["train"]
            except Exception as e2:
                self.logger.error(f"Fallback also failed: {e2}")
                raise e
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process Anthropic HH data into conversation format.
        
        Extracts conversations and formats them properly.
        """
        processed_data = []
        
        for item in raw_data:
            if "chosen" in item:
                # Use the chosen response (preferred by human raters)
                conversation = item["chosen"]
            else:
                # Fallback to OpenAssistant format if needed
                if item.get("role") == "assistant" and item.get("lang") == "en":
                    conversation = item["text"]
                else:
                    continue
            
            if not conversation or not isinstance(conversation, str):
                continue
                
            # Clean up the conversation
            conversation = conversation.strip()
            
            # Split into turns if it's a multi-turn conversation
            turns = self._extract_conversation_turns(conversation)
            
            for turn in turns:
                if len(turn.split()) >= self.config.min_length:
                    processed_data.append({
                        "text": turn,
                        "metadata": {
                            "source": "anthropic_hh",
                            "subset": self.subset,
                            "type": "conversation"
                        }
                    })
        
        return processed_data
    
    def _extract_conversation_turns(self, conversation: str) -> List[str]:
        """Extract individual turns from a conversation."""
        # Handle different conversation formats
        if "\n\nHuman:" in conversation and "\n\nAssistant:" in conversation:
            # Anthropic format
            turns = []
            parts = conversation.split("\n\nAssistant:")
            for part in parts[1:]:  # Skip first part (usually empty or human prompt)
                assistant_response = part.split("\n\nHuman:")[0].strip()
                if assistant_response:
                    turns.append(assistant_response)
            return turns
        else:
            # Single turn or different format
            return [conversation]
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply Anthropic HH-specific filters."""
        text = item["text"]
        
        # Filter out conversations that are too short
        word_count = len(text.split())
        if word_count < 10 or word_count > 400:
            return False
        
        # Filter out conversations with harmful content markers
        harmful_markers = ["I cannot", "I can't help", "I'm not able to", "I shouldn't"]
        if any(marker.lower() in text.lower() for marker in harmful_markers):
            return False
            
        # Keep helpful, informative responses
        return True
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single preprocessed Anthropic HH example."""
        try:
            # Handle preprocessed format (should have 'text' field)
            if "text" in example:
                text = example["text"].strip()
                
                # Basic quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                return {"text": text}
            else:
                # This shouldn't happen as preprocess should create 'text' field
                return None
                
        except Exception as e:
            return None

def create_anthropic_hh_loader(
    subset: str = "helpful-base", 
    max_samples: int = 30000, 
    max_length: int = 400
) -> AnthropicHHLoader:
    """Factory function to create Anthropic HH loader with custom config."""
    config = DatasetConfig(
        name=f"Anthropic-HH-{subset}",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return AnthropicHHLoader(config, subset)
