"""
LMSYS Chat Dataset Loader

Loads and processes LMSYS Chat-1M dataset for conversational training.
Real-world conversations with 25 state-of-the-art LLMs.
"""

from typing import List, Dict, Any, Optional
import json
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class LMSYSChatLoader(BaseDatasetLoader):
    """Loader for LMSYS Chat-1M dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="LMSYS-Chat-1M",
                max_samples=50000,
                max_length=1536,  # Increased from 512 for longer conversations
                min_length=50
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load LMSYS Chat-1M dataset from HuggingFace."""
        try:
            self.logger.info("Loading LMSYS Chat-1M dataset...")
            # Note: This dataset requires agreement to terms
            dataset = hf_load_dataset("lmsys/lmsys-chat-1m")
            
            if "train" in dataset:
                train_data = dataset["train"]
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                self.logger.info(f"Using split '{split_name}' from LMSYS Chat")
                train_data = dataset[split_name]
            
            # Sample from the dataset instead of returning the whole thing
            if self.config.shuffle:
                # Shuffle and take a sample
                shuffled = train_data.shuffle(seed=self.config.seed)
                sample_data = shuffled.select(range(min(self.config.max_samples, len(shuffled))))
            else:
                # Just take the first max_samples
                sample_data = train_data.select(range(min(self.config.max_samples, len(train_data))))
            
            # Convert to list of dicts for easier processing
            return [sample_data[i] for i in range(len(sample_data))]
                
        except Exception as e:
            self.logger.error(f"Failed to load LMSYS Chat dataset: {e}")
            self.logger.error("Note: This dataset requires accepting terms of use on HuggingFace")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """Preprocess LMSYS Chat-1M raw data into standardized format."""
        processed_data = []
        
        for example in raw_data:
            # Process the raw example directly into conversation format
            conversation_text = self._format_conversation(example)
            if conversation_text:
                processed_data.append({"text": conversation_text})
        
        return processed_data
    
    def _format_conversation(self, example: Dict[str, Any]) -> Optional[str]:
        """Convert LMSYS raw example into conversation text format."""
        try:
            # LMSYS format: conversation is already a list of message dicts
            conversation_data = example.get("conversation", [])
            model_name = example.get("model", "unknown")
            
            if not conversation_data or not isinstance(conversation_data, list):
                return None
            
            # Filter for high-quality models
            preferred_models = [
                "gpt-4", "gpt-3.5-turbo", "claude", "palm", "vicuna", 
                "alpaca", "chatglm", "koala", "llama"
            ]
            
            model_lower = model_name.lower()
            is_preferred = any(pref in model_lower for pref in preferred_models)
            
            # Process conversation
            conversation_parts = []
            for turn in conversation_data:
                role = turn.get("role", "").lower()
                content = turn.get("content", "").strip()
                
                if not content:
                    continue
                
                # Handle different role formats
                if role in ["user", "human"]:
                    conversation_parts.append(f"Human: {content}")
                elif role in ["assistant", "gpt", "ai"]:
                    conversation_parts.append(f"Assistant: {content}")
                elif role == "system":
                    # Include system messages if they're informative
                    if len(content) < 200 and "helpful assistant" not in content.lower():
                        conversation_parts.append(f"System: {content}")
            
            if len(conversation_parts) < 2:
                return None
            
            # Join conversation
            text = "\n\n".join(conversation_parts)
            
            # Quality checks
            if len(text) < self.config.min_length or len(text) > self.config.max_length:
                return None
            
            # Filter out conversations with PII redaction artifacts
            if "NAME_" in text and text.count("NAME_") > 5:  # Increased threshold
                return None
            
            # Filter out conversations flagged by moderation
            moderation_list = example.get("openai_moderation", [])
            if isinstance(moderation_list, list) and len(moderation_list) > 0:
                # Check if any turn is flagged
                any_flagged = any(mod.get("flagged", False) for mod in moderation_list if isinstance(mod, dict))
                if any_flagged:
                    return None
            
            # Language filter - prefer English
            language = example.get("language", "en")
            if language.lower() not in ["en", "english", "american english", "british english"]:
                return None
            
            # Filter conversations that are too repetitive
            if self._is_repetitive(text):
                return None
                
            return text
            
        except Exception as e:
            return None
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text is overly repetitive."""
        # Simple repetition check - look for repeated phrases
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Check for repeated lines
        line_counts = {}
        for line in lines:
            cleaned_line = line.strip().lower()
            if len(cleaned_line) > 10:  # Only check substantial lines
                line_counts[cleaned_line] = line_counts.get(cleaned_line, 0) + 1
        
        # If any line appears more than 3 times, consider it repetitive
        max_repeats = max(line_counts.values()) if line_counts else 0
        return max_repeats > 3
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single preprocessed LMSYS Chat example."""
        try:
            # Handle preprocessed format (should have 'text' field)
            if "text" in example:
                text = example["text"].strip()
                
                # Basic quality checks
                if len(text) < self.config.min_length or len(text) > self.config.max_length:
                    return None
                
                return {"text": text}
            else:
                # Fallback to format raw data if needed
                return {"text": self._format_conversation(example)} if self._format_conversation(example) else None
                
        except Exception as e:
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the LMSYS Chat dataset."""
        return {
            "name": "LMSYS Chat-1M",
            "description": "Real-world conversations with 25 state-of-the-art LLMs",
            "source": "lmsys/lmsys-chat-1m",
            "total_size": "1M conversations from 210K unique users",
            "quality": "High - Real user interactions",
            "use_case": "Natural conversation patterns, diverse user queries",
            "note": "Requires accepting terms of use on HuggingFace"
        }

def create_lmsys_chat_loader(max_samples: int = 30000, max_length: int = 6000) -> LMSYSChatLoader:
    """Factory function to create LMSYS Chat loader with custom config."""
    config = DatasetConfig(
        name="LMSYS-Chat-1M",
        max_samples=max_samples,
        max_length=max_length,  # Increased from 512 for longer conversations
        min_length=50
    )
    return LMSYSChatLoader(config)
