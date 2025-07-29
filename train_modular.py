#!/usr/bin/env python3
"""
Modular training script for transformer models.
This uses the new modular training system with separate modules for:
- Configuration and argument parsing
- Logging setup
- Model initialization
- Dataset loading
- Training callbacks
- Training loop orchestration
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import modular components
from training.core.config import parse_arguments, validate_config
from training.core.logging_setup import setup_logging
from training.core.model_setup import setup_tokenizer, create_model, setup_output_directory, save_model_config
from training.core.trainer_setup import setup_training_args, create_trainer
from training.core.training_loop import (
    handle_checkpoint_loading,
    setup_callbacks_for_architecture, 
    execute_training,
    save_final_model
)

# Import dataset loading utilities
from training.data.openassistant import load_openassistant_dataset
from training.data.knowledge_datasets import load_wikipedia_dataset, load_pile_dataset
from training.data.conversational import load_conversational_dataset
from training.utils.dataset_utils import combine_datasets, _print_split_stats

# Try to use modern loader system
try:
    from training.data.loader_manager import load_datasets_with_loader
    MODERN_LOADERS_AVAILABLE = True
except ImportError:
    MODERN_LOADERS_AVAILABLE = False

def load_datasets(tokenizer, args):
    """Load datasets based on configuration."""
    logger = logging.getLogger(__name__)
    
    # Try modern loader system first
    if MODERN_LOADERS_AVAILABLE:
        try:
            train_ds, val_ds = load_datasets_with_loader(tokenizer, args)
            if train_ds is not None and val_ds is not None:
                logger.info("Successfully loaded datasets using modern loader system")
                return train_ds, val_ds
        except Exception as e:
            logger.warning(f"Modern loader system failed: {e}, falling back to legacy loading")
    
    # Fallback to legacy dataset loading
    logger.info("Using legacy dataset loading system")
    
    # Load OpenAssistant dataset as base
    print("Loading OpenAssistant OASST1 dataset...")
    oa_train_ds, oa_val_ds = load_openassistant_dataset(tokenizer, max_len=args.max_length)
    
    datasets_to_combine = [(oa_train_ds, oa_val_ds)]
    dataset_parts = ["OpenAssistant"]
    
    # Optionally add conversational dataset
    if args.use_conversation:
        print("Loading conversational dataset for improved chat abilities...")
        conv_train_ds, conv_val_ds = load_conversational_dataset(
            tokenizer, 
            max_len=args.max_length, 
            max_samples=30000
        )
        if conv_train_ds is not None and conv_val_ds is not None:
            datasets_to_combine.append((conv_train_ds, conv_val_ds))
            dataset_parts.append("Conversational (HH-RLHF)")
            print("Successfully integrated conversational dataset!")
        else:
            print("Failed to load conversational dataset, continuing with existing datasets...")
    
    # Optionally add Wikipedia
    if args.use_wiki:
        print("Loading Wikipedia dataset...")
        wiki_train_ds, wiki_val_ds = load_wikipedia_dataset(tokenizer, max_len=args.max_length)
        datasets_to_combine.append((wiki_train_ds, wiki_val_ds))
        dataset_parts.append("Wikipedia")
    
    # Optionally add The Pile
    if args.use_pile:
        print("Loading The Pile dataset...")
        pile_train_ds, pile_val_ds = load_pile_dataset(tokenizer, max_len=args.max_length)
        datasets_to_combine.append((pile_train_ds, pile_val_ds))
        dataset_parts.append("The Pile")
    
    # Combine all datasets
    if len(datasets_to_combine) == 1:
        train_ds, val_ds = datasets_to_combine[0]
    else:
        train_ds, val_ds = datasets_to_combine[0]
        for additional_train, additional_val in datasets_to_combine[1:]:
            train_ds, val_ds = combine_datasets(
                (train_ds, val_ds),
                (additional_train, additional_val)
            )
    
    # Print final dataset stats
    dataset_description = " + ".join(dataset_parts)
    print(f"Using {dataset_description} dataset")
    _print_split_stats("Final Combined", train_ds, val_ds)
    
    return train_ds, val_ds

def main():
    """Main training function."""
    # Parse arguments and validate configuration
    args = parse_arguments()
    validate_config(args)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting modular transformer training...")
    logger.info(f"Architecture: {args.architecture}")
    if args.architecture == "dense":
        logger.info(f"Model size: {args.model_size}")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    logger.info("Tokenizer configured")
    
    # Create model
    model, cfg = create_model(args, tokenizer)
    logger.info("Model created successfully")
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    logger.info(f"Output directory: {output_dir}")
    
    # Save model configuration
    save_model_config(output_dir, cfg, args)
    
    # Load datasets
    train_ds, val_ds = load_datasets(tokenizer, args)
    logger.info(f"Datasets loaded - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Setup training arguments
    training_args = setup_training_args(args, output_dir)
    logger.info("Training arguments configured")
    
    # Handle checkpoint loading
    resume_from_checkpoint, model_weights_loaded = handle_checkpoint_loading(args, model, output_dir)
    
    if model_weights_loaded:
        logger.info("Model weights loaded from checkpoint, starting fresh training")
    elif resume_from_checkpoint:
        logger.info(f"Will resume training from: {resume_from_checkpoint}")
    else:
        logger.info("Starting completely fresh training")
    
    # Setup callbacks
    callbacks = setup_callbacks_for_architecture(args, training_args)
    logger.info("Training callbacks configured")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=callbacks,
        architecture=args.architecture
    )
    logger.info("Trainer created successfully")
    
    # Execute training
    execute_training(trainer, resume_from_checkpoint if not args.ignore_checkpoint else None)
    
    # Save final model
    final_model_path = save_final_model(trainer, output_dir)
    logger.info(f"Training completed! Final model saved to: {final_model_path}")
    
    return trainer, final_model_path

if __name__ == "__main__":
    try:
        trainer, final_model_path = main()
        print("\n🎉 Training completed successfully!")
        print(f"📁 Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
