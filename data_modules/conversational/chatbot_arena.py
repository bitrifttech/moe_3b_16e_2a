"""
Chatbot Arena Dataset Loader

Loads and processes Chatbot Arena conversations dataset for conversational training.
High-quality conversations with human preference rankings.
"""

from typing import List, Dict, Any
import json
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class ChatbotArenaLoader(BaseDatasetLoader):
    """Loader for Chatbot Arena conversations dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        if config is None:
            config = DatasetConfig(
                name="ChatbotArena",
                max_samples=30000,
                max_length=512,
                min_length=50
            )
        super().__init__(config)
    
    def load_raw_data(self) -> Any:
        """Load Chatbot Arena dataset from HuggingFace."""
        try:
            self.logger.info("Loading Chatbot Arena conversations dataset...")
            # Note: This dataset requires agreement to terms
            dataset = hf_load_dataset("lmsys/chatbot_arena_conversations")
            
            if "train" in dataset:
                return dataset["train"]
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                self.logger.info(f"Using split '{split_name}' from Chatbot Arena")
                return dataset[split_name]
                
        except Exception as e:
            self.logger.error(f"Failed to load Chatbot Arena dataset: {e}")
            self.logger.error("Note: This dataset requires accepting terms of use on HuggingFace")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """Preprocess Chatbot Arena raw data into standardized format."""
        processed_data = []
        
        for example in raw_data:
            processed_example = self.process_example(example)
            if processed_example is not None:
                processed_data.append(processed_example)
        
        return processed_data
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Process a single Chatbot Arena example into conversation format."""
        try:
            # Arena format has two model responses (A and B) with human preference
            conversation_a = example.get("conversation_a", [])
            conversation_b = example.get("conversation_b", [])
            winner = example.get("winner", "")
            
            # Parse conversations if they're strings
            if isinstance(conversation_a, str):
                try:
                    conversation_a = json.loads(conversation_a)
                except json.JSONDecodeError:
                    return None
                    
            if isinstance(conversation_b, str):
                try:
                    conversation_b = json.loads(conversation_b)
                except json.JSONDecodeError:
                    return None
            
            # Choose the preferred conversation based on human vote
            if winner == "model_a" and conversation_a:
                chosen_conversation = conversation_a
            elif winner == "model_b" and conversation_b:
                chosen_conversation = conversation_b
            elif winner == "tie" and conversation_a:
                # For ties, randomly pick one (or prefer A)
                chosen_conversation = conversation_a
            else:
                # If no clear winner or invalid data, skip
                return None
            
            if not chosen_conversation or not isinstance(chosen_conversation, list):
                return None
            
            # Process the chosen conversation
            conversation_parts = []
            for turn in chosen_conversation:
                role = turn.get("role", "").lower()
                content = turn.get("content", "").strip()
                
                if not content:
                    continue
                
                if role in ["user", "human"]:
                    conversation_parts.append(f"Human: {content}")
                elif role in ["assistant", "gpt", "ai"]:
                    conversation_parts.append(f"Assistant: {content}")
                elif role == "system":
                    # Include brief system messages
                    if len(content) < 150 and "helpful assistant" not in content.lower():
                        conversation_parts.append(f"System: {content}")
            
            if len(conversation_parts) < 2:
                return None
            
            # Join conversation
            text = "\n\n".join(conversation_parts)
            
            # Quality checks
            if len(text) < self.config.min_length or len(text) > self.config.max_length:
                return None
            
            # Filter out flagged content
            if example.get("openai_moderation", {}).get("flagged", False):
                return None
            
            # Filter out toxic content if flagged
            if example.get("toxic", False):
                return None
            
            # Language filter
            language = example.get("language", "en")
            if language != "en":
                return None
            
            return {"text": text}
            
        except Exception as e:
            self.logger.warning(f"Error processing Chatbot Arena example: {e}")
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the Chatbot Arena dataset."""
        return {
            "name": "Chatbot Arena Conversations",
            "description": "Conversations with human preference rankings from Chatbot Arena",
            "source": "lmsys/chatbot_arena_conversations",
            "total_size": "33K conversations with pairwise preferences",
            "quality": "Very High - Human-preferred responses",
            "use_case": "Learning from human-preferred conversation patterns",
            "note": "Requires accepting terms of use on HuggingFace"
        }

def create_chatbot_arena_loader(max_samples: int = 30000, max_length: int = 512) -> ChatbotArenaLoader:
    """Factory function to create Chatbot Arena loader with custom config."""
    config = DatasetConfig(
        name="ChatbotArena",
        max_samples=max_samples,
        max_length=max_length,
        min_length=50
    )
    return ChatbotArenaLoader(config)
