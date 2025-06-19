"""
Base Dataset Loader

Provides the abstract base class for all dataset loaders in the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset as hf_load_dataset
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    name: str
    max_samples: Optional[int] = None
    max_length: int = 512
    min_length: int = 10
    train_split: float = 0.95
    validation_split: float = 0.05
    shuffle: bool = True
    seed: int = 42
    custom_filters: Optional[Dict[str, Any]] = None
    preprocessing_options: Optional[Dict[str, Any]] = None

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    
    Provides common functionality and interface for loading, processing,
    and preparing datasets for LLM training.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.name = config.name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def load_raw_data(self) -> Any:
        """Load the raw dataset from source (HuggingFace, local files, etc.)"""
        pass
    
    @abstractmethod
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Preprocess raw data into standardized format.
        
        Returns:
            List of dicts with keys: 'text', 'metadata' (optional)
        """
        pass
    
    def filter_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply length and custom filters to the data."""
        filtered_data = []
        
        for item in data:
            text = item.get('text', '')
            
            # Length filtering
            if len(text.split()) < self.config.min_length:
                continue
            if len(text.split()) > self.config.max_length:
                # Truncate instead of dropping
                words = text.split()[:self.config.max_length]
                item['text'] = ' '.join(words)
            
            # Custom filters
            if self.config.custom_filters:
                if not self._apply_custom_filters(item):
                    continue
                    
            filtered_data.append(item)
            
            # Max samples limit
            if self.config.max_samples and len(filtered_data) >= self.config.max_samples:
                break
                
        return filtered_data
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply custom filtering logic. Override in subclasses if needed."""
        return True
    
    def split_data(self, data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Split data into train and validation sets."""
        if self.config.shuffle:
            import random
            random.seed(self.config.seed)
            data = data.copy()
            random.shuffle(data)
        
        train_size = int(len(data) * self.config.train_split)
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        return train_data, val_data
    
    def get_stats(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not data:
            return {"total_samples": 0}
            
        total_samples = len(data)
        total_words = sum(len(item['text'].split()) for item in data)
        avg_words = total_words / total_samples if total_samples > 0 else 0
        
        word_counts = [len(item['text'].split()) for item in data]
        min_words = min(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        
        return {
            "total_samples": total_samples,
            "total_words": total_words,
            "avg_words_per_sample": round(avg_words, 2),
            "min_words": min_words,
            "max_words": max_words,
            "estimated_size_mb": round(total_words * 6 / 1024 / 1024, 2)  # Rough estimate
        }
    
    def load_and_process(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
        """
        Complete pipeline: load, preprocess, filter, and split data.
        
        Returns:
            Tuple of (train_data, val_data, stats)
        """
        self.logger.info(f"Loading {self.name} dataset...")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess
        self.logger.info(f"Preprocessing {self.name} dataset...")
        processed_data = self.preprocess(raw_data)
        
        # Filter
        self.logger.info(f"Filtering {self.name} dataset...")
        filtered_data = self.filter_data(processed_data)
        
        # Split
        train_data, val_data = self.split_data(filtered_data)
        
        # Get statistics
        stats = self.get_stats(filtered_data)
        stats['train_samples'] = len(train_data)
        stats['val_samples'] = len(val_data)
        
        self.logger.info(f"{self.name} - Train: {len(train_data)}, Val: {len(val_data)}, Total: {len(filtered_data)}")
        
        return train_data, val_data, stats
