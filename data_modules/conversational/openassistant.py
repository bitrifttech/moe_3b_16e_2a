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
        Process OpenAssistant data into proper instruction-response format.
        
        Builds conversation threads with human-assistant context.
        """
        processed_data = []
        
        # Get the training split
        train_data = raw_data["train"]
        
        # Build conversation threads
        conversations = self._build_conversation_threads(train_data)
        
        for conversation in conversations:
            if len(conversation) >= 2:  # Need at least human + assistant
                formatted_text = self._format_conversation_thread(conversation)
                if formatted_text:
                    processed_data.append({
                        "text": formatted_text,
                        "metadata": {
                            "source": "openassistant",
                            "conversation_id": conversation[0].get("conversation_id", ""),
                            "quality": max(msg.get("rank", 0) for msg in conversation),
                            "turns": len(conversation)
                        }
                    })
        
        return processed_data
    
    def _build_conversation_threads(self, data) -> List[List[Dict]]:
        """Build conversation threads from OpenAssistant message tree."""
        # Group messages by conversation
        conversations = {}
        
        for message in data:
            if message.get("lang") != "en":
                continue
                
            conv_id = message.get("conversation_id")
            if not conv_id:
                continue
                
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(message)
        
        # Build threaded conversations
        threaded_conversations = []
        
        for conv_id, messages in conversations.items():
            # Sort by message tree structure
            thread = self._build_message_thread(messages)
            if thread:
                threaded_conversations.append(thread)
        
        return threaded_conversations
    
    def _build_message_thread(self, messages: List[Dict]) -> List[Dict]:
        """Build a single conversation thread from messages."""
        # Create a mapping of message_id to message
        msg_map = {msg["message_id"]: msg for msg in messages}
        
        # Find the root message (no parent)
        root_messages = [msg for msg in messages if not msg.get("parent_id")]
        if not root_messages:
            return []
        
        # Build thread starting from root
        thread = []
        current_msg = root_messages[0]  # Take first root
        
        while current_msg:
            thread.append(current_msg)
            
            # Find children of current message
            children = [msg for msg in messages if msg.get("parent_id") == current_msg["message_id"]]
            
            if children:
                # Take the highest ranked child (best response)
                current_msg = max(children, key=lambda x: x.get("rank", 0))
            else:
                current_msg = None
        
        return thread
    
    def _format_conversation_thread(self, thread: List[Dict]) -> str:
        """Format conversation thread as instruction-response pairs."""
        formatted_parts = []
        
        for i, message in enumerate(thread):
            role = message.get("role", "")
            text = message.get("text", "").strip()
            
            if not text:
                continue
            
            if role == "prompter":  # Human/user message
                formatted_parts.append(f"Human: {text}")
            elif role == "assistant":  # Assistant response
                formatted_parts.append(f"Assistant: {text}")
        
        if len(formatted_parts) >= 2:  # Need at least one exchange
            return "\n\n".join(formatted_parts)
        
        return ""
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply OpenAssistant-specific filters."""
        text = item["text"]
        
        # Filter out very short conversations
        word_count = len(text.split())
        if word_count < 20 or word_count > 1000:  # Increased limits for full conversations
            return False
        
        # Ensure we have proper conversation structure
        if "Human:" not in text and "Assistant:" not in text:
            return False
            
        # Filter out responses that are mostly code (basic heuristic)
        code_indicators = ["```", "def ", "class ", "import ", "function(", "SELECT ", "FROM "]
        code_ratio = sum(1 for indicator in code_indicators if indicator in text) / len(code_indicators)
        if code_ratio > 0.4:  # If more than 40% code indicators
            return False
        
        # Ensure conversation has both human and assistant parts
        has_human = "Human:" in text
        has_assistant = "Assistant:" in text
        
        return has_human and has_assistant

def create_openassistant_loader(max_samples: int = 25000, max_length: int = 400) -> OpenAssistantLoader:
    """Factory function to create OpenAssistant loader with custom config."""
    config = DatasetConfig(
        name="OpenAssistant",
        max_samples=max_samples,
        max_length=max_length,
        min_length=20
    )
    return OpenAssistantLoader(config)
