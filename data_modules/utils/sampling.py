"""
Smart Sampling Utilities

Provides intelligent sampling strategies for dataset preparation.
"""

from typing import List, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

class SmartSampler:
    """
    Intelligent sampling strategies for dataset preparation.
    
    Supports various sampling methods:
    - Quality-based: Sample based on quality metrics
    - Length-balanced: Ensure good distribution of text lengths
    - Diversity: Maximize content diversity
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.logger = logging.getLogger(f"{__name__}.SmartSampler")
    
    def quality_sample(
        self,
        data: List[Dict[str, str]],
        target_count: int,
        quality_key: str = "quality",
        min_quality: float = 0.5
    ) -> List[Dict[str, str]]:
        """
        Sample based on quality scores.
        
        Args:
            data: Dataset to sample from
            target_count: Number of samples to select
            quality_key: Key in metadata containing quality score
            min_quality: Minimum quality threshold
        """
        # Filter by minimum quality
        quality_filtered = []
        for item in data:
            quality = item.get("metadata", {}).get(quality_key, 1.0)
            if isinstance(quality, (int, float)) and quality >= min_quality:
                quality_filtered.append((item, quality))
        
        if not quality_filtered:
            self.logger.warning("No items meet quality threshold, using random sampling")
            return self.random_sample(data, target_count)
        
        # Sort by quality (descending)
        quality_filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Take top quality items
        selected_count = min(target_count, len(quality_filtered))
        selected = [item for item, _ in quality_filtered[:selected_count]]
        
        self.logger.info(f"Quality sampling: selected {len(selected)} items with quality >= {min_quality}")
        return selected
    
    def length_balanced_sample(
        self,
        data: List[Dict[str, str]],
        target_count: int,
        length_bins: int = 5
    ) -> List[Dict[str, str]]:
        """
        Sample to ensure balanced distribution of text lengths.
        
        Args:
            data: Dataset to sample from
            target_count: Number of samples to select
            length_bins: Number of length bins to create
        """
        # Calculate text lengths
        items_with_length = [(item, len(item["text"].split())) for item in data]
        
        # Create length bins
        lengths = [length for _, length in items_with_length]
        min_len, max_len = min(lengths), max(lengths)
        bin_size = (max_len - min_len) / length_bins
        
        bins = [[] for _ in range(length_bins)]
        
        for item, length in items_with_length:
            bin_idx = min(int((length - min_len) / bin_size), length_bins - 1)
            bins[bin_idx].append(item)
        
        # Sample proportionally from each bin
        samples_per_bin = target_count // length_bins
        remainder = target_count % length_bins
        
        selected = []
        for i, bin_items in enumerate(bins):
            if not bin_items:
                continue
                
            bin_target = samples_per_bin + (1 if i < remainder else 0)
            bin_sample = random.sample(bin_items, min(bin_target, len(bin_items)))
            selected.extend(bin_sample)
        
        # If we're short, fill from largest bins
        if len(selected) < target_count:
            remaining_items = [item for bin_items in bins for item in bin_items if item not in selected]
            additional = random.sample(remaining_items, min(target_count - len(selected), len(remaining_items)))
            selected.extend(additional)
        
        self.logger.info(f"Length-balanced sampling: selected {len(selected)} items across {length_bins} bins")
        return selected
    
    def diversity_sample(
        self,
        data: List[Dict[str, str]],
        target_count: int,
        diversity_key: str = "source"
    ) -> List[Dict[str, str]]:
        """
        Sample to maximize diversity based on a metadata key.
        
        Args:
            data: Dataset to sample from
            target_count: Number of samples to select
            diversity_key: Metadata key to use for diversity (e.g., 'source', 'type')
        """
        # Group by diversity key
        groups = {}
        for item in data:
            key_value = item.get("metadata", {}).get(diversity_key, "unknown")
            if key_value not in groups:
                groups[key_value] = []
            groups[key_value].append(item)
        
        # Sample proportionally from each group
        total_groups = len(groups)
        samples_per_group = target_count // total_groups
        remainder = target_count % total_groups
        
        selected = []
        group_names = list(groups.keys())
        
        for i, group_name in enumerate(group_names):
            group_items = groups[group_name]
            group_target = samples_per_group + (1 if i < remainder else 0)
            group_sample = random.sample(group_items, min(group_target, len(group_items)))
            selected.extend(group_sample)
        
        self.logger.info(f"Diversity sampling: selected {len(selected)} items from {total_groups} groups")
        return selected
    
    def random_sample(
        self,
        data: List[Dict[str, str]],
        target_count: int
    ) -> List[Dict[str, str]]:
        """Simple random sampling."""
        if target_count >= len(data):
            return data.copy()
        
        selected = random.sample(data, target_count)
        self.logger.info(f"Random sampling: selected {len(selected)} items")
        return selected
    
    def hybrid_sample(
        self,
        data: List[Dict[str, str]],
        target_count: int,
        quality_weight: float = 0.4,
        length_weight: float = 0.3,
        diversity_weight: float = 0.3,
        quality_key: str = "quality",
        diversity_key: str = "source"
    ) -> List[Dict[str, str]]:
        """
        Hybrid sampling combining multiple strategies.
        
        Args:
            data: Dataset to sample from
            target_count: Number of samples to select
            quality_weight: Weight for quality-based sampling
            length_weight: Weight for length-balanced sampling
            diversity_weight: Weight for diversity sampling
        """
        # Normalize weights
        total_weight = quality_weight + length_weight + diversity_weight
        quality_weight /= total_weight
        length_weight /= total_weight
        diversity_weight /= total_weight
        
        # Calculate samples for each strategy
        quality_samples = int(target_count * quality_weight)
        length_samples = int(target_count * length_weight)
        diversity_samples = target_count - quality_samples - length_samples
        
        # Apply each sampling strategy
        selected_items = set()
        
        # Quality sampling
        if quality_samples > 0:
            quality_selected = self.quality_sample(data, quality_samples, quality_key)
            selected_items.update(id(item) for item in quality_selected)
        
        # Length-balanced sampling (from remaining items)
        remaining_data = [item for item in data if id(item) not in selected_items]
        if length_samples > 0 and remaining_data:
            length_selected = self.length_balanced_sample(remaining_data, length_samples)
            selected_items.update(id(item) for item in length_selected)
        
        # Diversity sampling (from remaining items)
        remaining_data = [item for item in data if id(item) not in selected_items]
        if diversity_samples > 0 and remaining_data:
            diversity_selected = self.diversity_sample(remaining_data, diversity_samples, diversity_key)
            selected_items.update(id(item) for item in diversity_selected)
        
        # Collect final selection
        final_selection = [item for item in data if id(item) in selected_items]
        
        self.logger.info(f"Hybrid sampling: selected {len(final_selection)} items using combined strategies")
        return final_selection
