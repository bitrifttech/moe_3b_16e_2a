"""
Modern modular dataset system for loading and managing multiple datasets.
"""
import logging
import sys
import os

# Add parent directory to path for dataset loaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import dataset loaders from the modular system
try:
    from datasets.loader_registry import (
        create_openassistant_loader,
        create_anthropic_hh_loader,
        create_sharegpt_loader,
        create_alpaca_loader,
        create_wikipedia_loader,
        create_the_pile_loader,
        create_ultrachat_loader,
        create_openorca_loader,
        create_lmsys_chat_loader,
        create_chatbot_arena_loader,
    )
    MODERN_LOADERS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Modern dataset loaders not available: {e}")
    MODERN_LOADERS_AVAILABLE = False

def load_datasets_with_loader(tokenizer, args):
    """Load datasets using the new modular system."""
    if not MODERN_LOADERS_AVAILABLE:
        logger.error("Modern dataset loaders not available, falling back to legacy loading")
        return None, None
    
    datasets = []
    logger = logging.getLogger(__name__)
    
    try:
        # Load OpenAssistant if not using conversation dataset
        if not args.use_conversation:
            oa_loader = create_openassistant_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            oa_train, oa_val, _ = oa_loader.load_and_process()
            datasets.append(("openassistant", oa_train, oa_val))
        
        # Load Anthropic HH if enabled
        if args.use_conversation:
            hh_loader = create_anthropic_hh_loader(
                subset="helpful-base",
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            hh_train, hh_val, _ = hh_loader.load_and_process()
            datasets.append(("anthropic_hh", hh_train, hh_val))
        
        # Load ShareGPT if enabled
        if args.use_sharegpt:
            try:
                sgpt_loader = create_sharegpt_loader(
                    max_samples=args.max_samples,
                    max_length=args.max_length
                )
                sgpt_train, sgpt_val, _ = sgpt_loader.load_and_process()
                datasets.append(("sharegpt", sgpt_train, sgpt_val))
            except Exception as e:
                logger.warning(f"Failed to load ShareGPT: {e}")
        
        # Load Alpaca if enabled
        if args.use_alpaca:
            alpaca_loader = create_alpaca_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            alpaca_train, alpaca_val, _ = alpaca_loader.load_and_process()
            datasets.append(("alpaca", alpaca_train, alpaca_val))
        
        # Load Wikipedia if enabled
        if args.use_wiki:
            wiki_loader = create_wikipedia_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            wiki_train, wiki_val, _ = wiki_loader.load_and_process()
            datasets.append(("wikipedia", wiki_train, wiki_val))
        
        # Load The Pile if enabled
        if args.use_pile:
            pile_loader = create_the_pile_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            pile_train, pile_val, _ = pile_loader.load_and_process()
            datasets.append(("the_pile", pile_train, pile_val))
        
        # Load UltraChat if enabled
        if args.use_ultrachat:
            ultrachat_loader = create_ultrachat_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            ultrachat_train, ultrachat_val, _ = ultrachat_loader.load_and_process()
            datasets.append(("ultrachat", ultrachat_train, ultrachat_val))
        
        # Load OpenOrca if enabled
        if args.use_openorca:
            openorca_loader = create_openorca_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            openorca_train, openorca_val, _ = openorca_loader.load_and_process()
            datasets.append(("openorca", openorca_train, openorca_val))
        
        # Load LMSYS Chat-1M if enabled
        if args.use_lmsys_chat:
            lmsys_chat_loader = create_lmsys_chat_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            lmsys_chat_train, lmsys_chat_val, _ = lmsys_chat_loader.load_and_process()
            datasets.append(("lmsys_chat", lmsys_chat_train, lmsys_chat_val))
        
        # Load Chatbot Arena if enabled
        if args.use_chatbot_arena:
            chatbot_arena_loader = create_chatbot_arena_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            chatbot_arena_train, chatbot_arena_val, _ = chatbot_arena_loader.load_and_process()
            datasets.append(("chatbot_arena", chatbot_arena_train, chatbot_arena_val))
        
        # If no datasets were loaded, use OpenAssistant as default
        if not datasets:
            oa_loader = create_openassistant_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            oa_train, oa_val, _ = oa_loader.load_and_process()
            datasets.append(("openassistant", oa_train, oa_val))
        
        return _combine_datasets(datasets)
        
    except Exception as e:
        logger.error(f"Error in dataset loading: {e}")
        return None, None


def _combine_datasets(datasets):
    """Combine multiple datasets into train/val splits."""
    logger = logging.getLogger(__name__)
    
    if len(datasets) == 1:
        return datasets[0][1], datasets[0][2]  # Return single dataset as is
    
    # For multiple datasets, just concatenate them (no complex mixing)
    all_train = []
    all_val = []
    
    for name, train, val in datasets:
        all_train.extend(train)
        all_val.extend(val)
        logger.info(f"{name} - Train: {len(train)}, Val: {len(val)}")
    
    logger.info(f"Combined dataset - Train: {len(all_train)}, Val: {len(all_val)}")
    return all_train, all_val
