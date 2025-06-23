"""
LMSYS Chat Dataset Loader

Loads and processes LMSYS Chat-1M dataset for conversational training.
Real-world conversations with 25 state-of-the-art LLMs.
"""

from typing import List, Dict, Any
import json
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class LMSYSChatLoader(BaseDatasetLoader):
    """Loader for LMSYS Chat-1M dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="LMSYS-Chat",
                max_samples=50000,
                max_length=512,
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
                return dataset["train"]
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                self.logger.info(f"Using split '{split_name}' from LMSYS Chat")
                return dataset[split_name]
                
        except Exception as e:
            self.logger.error(f"Failed to load LMSYS Chat dataset: {e}")
            self.logger.error("Note: This dataset requires accepting terms of use on HuggingFace")
            raise
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Process a single LMSYS Chat example into conversation format."""
        try:
            # LMSYS format: conversation is in JSON format
            conversation_data = example.get("conversation", [])
            model_name = example.get("model", "unknown")
            
            if isinstance(conversation_data, str):
                try:
                    conversation_data = json.loads(conversation_data)
                except json.JSONDecodeError:
                    return None
            
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
            if "NAME_" in text and text.count("NAME_") > 3:
                return None
            
            # Filter out conversations flagged by moderation
            if example.get("openai_moderation", {}).get("flagged", False):
                return None
            
            # Language filter - prefer English
            language = example.get("language", "en")
            if language != "en":
                return None
            
            return {"text": text}
            
        except Exception as e:
            self.logger.warning(f"Error processing LMSYS Chat example: {e}")
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

def create_lmsys_chat_loader(max_samples: int = 50000, max_length: int = 512) -> LMSYSChatLoader:
    """Factory function to create LMSYS Chat loader with custom config."""
    config = DatasetConfig(
        name="LMSYS-Chat",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return LMSYSChatLoader(config)
