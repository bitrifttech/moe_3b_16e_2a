"""
Dataset Mixing Utilities

Provides functionality to combine multiple datasets with configurable mixing ratios.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import logging
from ..base import DatasetConfig

logger = logging.getLogger(__name__)

class DatasetMixer:
    """
    Combines multiple datasets with specified mixing ratios.
    
    Supports various mixing strategies:
    - Proportional: Mix datasets based on their relative sizes
    - Equal: Equal representation from each dataset
    - Custom: User-specified ratios
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.logger = logging.getLogger(f"{__name__}.DatasetMixer")
    
    def mix_datasets(
        self,
        datasets: List[Tuple[List[Dict[str, str]], str]],  # (data, name)
        mixing_strategy: str = "proportional",
        custom_ratios: Optional[Dict[str, float]] = None,
        total_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Mix multiple datasets according to the specified strategy.
        
        Args:
            datasets: List of (dataset, name) tuples
            mixing_strategy: "proportional", "equal", or "custom"
            custom_ratios: Dict of {dataset_name: ratio} for custom strategy
            total_samples: Target total number of samples (None = use all)
            
        Returns:
            Tuple of (mixed_data, mixing_stats)
        """
        if not datasets:
            return [], {}
        
        self.logger.info(f"Mixing {len(datasets)} datasets using {mixing_strategy} strategy")
        
        # Calculate mixing ratios
        ratios = self._calculate_ratios(datasets, mixing_strategy, custom_ratios)
        
        # Determine sample counts for each dataset
        sample_counts = self._calculate_sample_counts(datasets, ratios, total_samples)
        
        # Sample from each dataset
        mixed_data = []
        mixing_stats = {"datasets": {}, "total_samples": 0, "mixing_ratios": ratios}
        
        for i, (data, name) in enumerate(datasets):
            target_count = sample_counts[i]
            sampled_data = self._sample_from_dataset(data, target_count, name)
            
            mixed_data.extend(sampled_data)
            mixing_stats["datasets"][name] = {
                "original_size": len(data),
                "sampled_size": len(sampled_data),
                "ratio": ratios.get(name, 0)
            }
        
        # Shuffle the mixed dataset
        random.shuffle(mixed_data)
        mixing_stats["total_samples"] = len(mixed_data)
        
        self.logger.info(f"Mixed dataset created with {len(mixed_data)} total samples")
        for name, stats in mixing_stats["datasets"].items():
            self.logger.info(f"  {name}: {stats['sampled_size']} samples ({stats['ratio']:.1%})")
        
        return mixed_data, mixing_stats
    
    def _calculate_ratios(
        self,
        datasets: List[Tuple[List[Dict[str, str]], str]],
        strategy: str,
        custom_ratios: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate mixing ratios based on strategy."""
        dataset_names = [name for _, name in datasets]
        
        if strategy == "equal":
            ratio = 1.0 / len(datasets)
            return {name: ratio for name in dataset_names}
        
        elif strategy == "proportional":
            total_size = sum(len(data) for data, _ in datasets)
            ratios = {}
            for data, name in datasets:
                ratios[name] = len(data) / total_size
            return ratios
        
        elif strategy == "custom":
            if not custom_ratios:
                raise ValueError("Custom ratios must be provided for custom strategy")
            
            # Normalize ratios to sum to 1.0
            total_ratio = sum(custom_ratios.values())
            return {name: ratio / total_ratio for name, ratio in custom_ratios.items()}
        
        else:
            raise ValueError(f"Unknown mixing strategy: {strategy}")
    
    def _calculate_sample_counts(
        self,
        datasets: List[Tuple[List[Dict[str, str]], str]],
        ratios: Dict[str, float],
        total_samples: Optional[int]
    ) -> List[int]:
        """Calculate how many samples to take from each dataset."""
        if total_samples is None:
            # Use proportional to smallest dataset to avoid over-sampling
            min_effective_size = min(
                len(data) / ratios.get(name, 1.0) 
                for data, name in datasets
            )
            total_samples = int(min_effective_size)
        
        sample_counts = []
        for data, name in datasets:
            target_count = int(total_samples * ratios.get(name, 0))
            # Don't exceed available data
            actual_count = min(target_count, len(data))
            sample_counts.append(actual_count)
        
        return sample_counts
    
    def _sample_from_dataset(
        self,
        data: List[Dict[str, str]],
        target_count: int,
        dataset_name: str
    ) -> List[Dict[str, str]]:
        """Sample specified number of items from a dataset."""
        if target_count >= len(data):
            # Use all data
            sampled = data.copy()
        else:
            # Random sample
            sampled = random.sample(data, target_count)
        
        # Add dataset source to metadata
        for item in sampled:
            if "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["mixed_from"] = dataset_name
        
        return sampled

def create_conversational_mix(
    openassistant_data: List[Dict[str, str]],
    anthropic_data: List[Dict[str, str]],
    sharegpt_data: Optional[List[Dict[str, str]]] = None,
    alpaca_data: Optional[List[Dict[str, str]]] = None,
    total_samples: int = 50000
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Create a balanced mix optimized for conversational training.
    
    Prioritizes high-quality conversational datasets.
    """
    mixer = DatasetMixer()
    
    datasets = [
        (openassistant_data, "OpenAssistant"),
        (anthropic_data, "Anthropic-HH")
    ]
    
    # Add optional datasets if provided
    if sharegpt_data:
        datasets.append((sharegpt_data, "ShareGPT"))
    if alpaca_data:
        datasets.append((alpaca_data, "Alpaca"))
    
    # Custom ratios favoring high-quality conversation data
    if len(datasets) == 2:
        ratios = {"OpenAssistant": 0.4, "Anthropic-HH": 0.6}
    elif len(datasets) == 3:
        ratios = {"OpenAssistant": 0.3, "Anthropic-HH": 0.4, "ShareGPT": 0.3}
    else:
        ratios = {"OpenAssistant": 0.25, "Anthropic-HH": 0.35, "ShareGPT": 0.25, "Alpaca": 0.15}
    
    return mixer.mix_datasets(
        datasets,
        mixing_strategy="custom",
        custom_ratios=ratios,
        total_samples=total_samples
    )
