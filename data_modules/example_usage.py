"""
Example Usage of the Modular Dataset System

Demonstrates how to use the dataset loaders for conversational chatbot training.
"""

import logging
from datasets.conversational import (
    create_openassistant_loader,
    create_anthropic_hh_loader,
    create_sharegpt_loader,
    create_alpaca_loader
)
from datasets.knowledge import (
    create_wikipedia_loader,
    create_the_pile_loader
)
from datasets.utils import create_conversational_mix, DatasetMixer, SmartSampler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_conversational_datasets():
    """Example: Load all conversational datasets."""
    logger.info("Loading conversational datasets...")
    
    # Load OpenAssistant
    oa_loader = create_openassistant_loader(max_samples=15000, max_length=400)
    oa_train, oa_val, oa_stats = oa_loader.load_and_process()
    
    # Load Anthropic HH-RLHF
    hh_loader = create_anthropic_hh_loader(subset="helpful-base", max_samples=20000, max_length=400)
    hh_train, hh_val, hh_stats = hh_loader.load_and_process()
    
    # Load ShareGPT (optional)
    try:
        sgpt_loader = create_sharegpt_loader(max_samples=10000, max_length=500)
        sgpt_train, sgpt_val, sgpt_stats = sgpt_loader.load_and_process()
    except Exception as e:
        logger.warning(f"ShareGPT failed to load: {e}")
        sgpt_train, sgpt_val, sgpt_stats = [], [], {}
    
    # Load Alpaca
    alpaca_loader = create_alpaca_loader(max_samples=8000, max_length=300)
    alpaca_train, alpaca_val, alpaca_stats = alpaca_loader.load_and_process()
    
    return {
        "openassistant": (oa_train, oa_val, oa_stats),
        "anthropic_hh": (hh_train, hh_val, hh_stats),
        "sharegpt": (sgpt_train, sgpt_val, sgpt_stats),
        "alpaca": (alpaca_train, alpaca_val, alpaca_stats)
    }

def create_chatbot_training_mix():
    """Example: Create optimized mix for chatbot training."""
    logger.info("Creating chatbot training mix...")
    
    # Load datasets
    datasets = load_conversational_datasets()
    
    # Extract training data
    oa_train = datasets["openassistant"][0]
    hh_train = datasets["anthropic_hh"][0]
    sgpt_train = datasets["sharegpt"][0] if datasets["sharegpt"][0] else None
    alpaca_train = datasets["alpaca"][0]
    
    # Create conversational mix
    mixed_data, mix_stats = create_conversational_mix(
        openassistant_data=oa_train,
        anthropic_data=hh_train,
        sharegpt_data=sgpt_train,
        alpaca_data=alpaca_train,
        total_samples=40000
    )
    
    logger.info(f"Created mixed dataset with {len(mixed_data)} samples")
    logger.info(f"Mix statistics: {mix_stats}")
    
    return mixed_data, mix_stats

def add_knowledge_datasets(conversational_data):
    """Example: Add knowledge datasets to conversational mix."""
    logger.info("Adding knowledge datasets...")
    
    # Load Wikipedia
    wiki_loader = create_wikipedia_loader(max_samples=15000, max_length=600)
    wiki_train, _, wiki_stats = wiki_loader.load_and_process()
    
    # Load The Pile (smaller subset)
    try:
        pile_loader = create_the_pile_loader(max_samples=10000, max_length=800)
        pile_train, _, pile_stats = pile_loader.load_and_process()
    except Exception as e:
        logger.warning(f"The Pile failed to load: {e}")
        pile_train, pile_stats = [], {}
    
    # Mix with conversational data
    mixer = DatasetMixer()
    
    datasets = [
        (conversational_data, "Conversational"),
        (wiki_train, "Wikipedia")
    ]
    
    if pile_train:
        datasets.append((pile_train, "The-Pile"))
    
    # Custom ratios: 60% conversational, 30% wikipedia, 10% pile
    ratios = {"Conversational": 0.6, "Wikipedia": 0.3}
    if pile_train:
        ratios["The-Pile"] = 0.1
        ratios["Conversational"] = 0.6
        ratios["Wikipedia"] = 0.3
    
    final_mix, final_stats = mixer.mix_datasets(
        datasets,
        mixing_strategy="custom",
        custom_ratios=ratios,
        total_samples=60000
    )
    
    logger.info(f"Final mixed dataset: {len(final_mix)} samples")
    return final_mix, final_stats

def smart_sampling_example(data):
    """Example: Use smart sampling strategies."""
    logger.info("Demonstrating smart sampling...")
    
    sampler = SmartSampler()
    
    # Quality-based sampling
    quality_sample = sampler.quality_sample(data, target_count=1000)
    logger.info(f"Quality sample: {len(quality_sample)} items")
    
    # Length-balanced sampling
    length_sample = sampler.length_balanced_sample(data, target_count=1000)
    logger.info(f"Length-balanced sample: {len(length_sample)} items")
    
    # Diversity sampling
    diversity_sample = sampler.diversity_sample(data, target_count=1000, diversity_key="source")
    logger.info(f"Diversity sample: {len(diversity_sample)} items")
    
    # Hybrid sampling
    hybrid_sample = sampler.hybrid_sample(data, target_count=1000)
    logger.info(f"Hybrid sample: {len(hybrid_sample)} items")
    
    return hybrid_sample

def main():
    """Main example workflow."""
    logger.info("🚀 Starting dataset system example...")
    
    # Step 1: Create conversational mix
    conversational_mix, conv_stats = create_chatbot_training_mix()
    
    # Step 2: Add knowledge datasets
    final_dataset, final_stats = add_knowledge_datasets(conversational_mix)
    
    # Step 3: Apply smart sampling
    optimized_dataset = smart_sampling_example(final_dataset)
    
    logger.info("✅ Dataset preparation complete!")
    logger.info(f"Final dataset size: {len(optimized_dataset)} samples")
    
    # Example: Save dataset statistics
    import json
    stats_summary = {
        "conversational_stats": conv_stats,
        "final_mix_stats": final_stats,
        "optimized_size": len(optimized_dataset)
    }
    
    with open("dataset_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)
    
    logger.info("📊 Statistics saved to dataset_stats.json")
    
    return optimized_dataset

if __name__ == "__main__":
    # Run the example
    dataset = main()
    print(f"\n🎉 Generated dataset with {len(dataset)} samples ready for training!")
