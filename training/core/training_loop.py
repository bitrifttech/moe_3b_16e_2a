"""
Main training loop execution and checkpoint management.
"""
import os
import logging
from safetensors.torch import load_file

# Get logger
logger = logging.getLogger(__name__)

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.isdir(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    
    # Sort by step number
    def extract_step(checkpoint_name):
        try:
            return int(checkpoint_name.split('-')[-1])
        except (ValueError, IndexError):
            return 0
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    return os.path.join(output_dir, latest_checkpoint)

def handle_checkpoint_loading(args, model, output_dir):
    """Handle checkpoint loading based on arguments."""
    latest_checkpoint = find_latest_checkpoint(output_dir)
    resume_from_checkpoint = None
    model_weights_loaded = False
    
    if latest_checkpoint and not args.ignore_checkpoint:
        resume_from_checkpoint = latest_checkpoint
        logger.info(f"Found existing checkpoint: {resume_from_checkpoint}")
    elif latest_checkpoint and args.ignore_checkpoint:
        # Load model weights but start from step 1
        logger.info(f"Loading model weights from {latest_checkpoint} but starting training from step 1")
        try:
            safetensors_path = os.path.join(latest_checkpoint, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
                model_weights_loaded = True
                logger.info("Successfully loaded model weights from checkpoint")
            else:
                logger.warning(f"Safetensors file not found at {safetensors_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
    else:
        if args.ignore_checkpoint:
            logger.info("Ignoring existing checkpoints due to --ignore_checkpoint flag")
        else:
            logger.info(f"No checkpoints found in {output_dir}")
    
    return resume_from_checkpoint, model_weights_loaded

def setup_callbacks_for_architecture(args, training_args):
    """Setup callbacks based on architecture."""
    from ..callbacks.progress import ProgressCallback
    from ..callbacks.checkpoint import CheckpointInfoCallback
    from ..callbacks.optimization import LrKickOnceCallback, MoERoutingCallback
    
    progress_callback = ProgressCallback(training_args.max_steps)
    callbacks = [progress_callback, CheckpointInfoCallback()]
    
    if args.architecture == "moe":
        # Add MoE-specific callbacks
        lr_kick_callback = LrKickOnceCallback(factor=5.0, at_step=44500)  # Re-enabled to break plateau
        moe_routing_callback = MoERoutingCallback(log_every=5)
        callbacks.extend([lr_kick_callback, moe_routing_callback])
        print("Added MoE-specific callbacks: LR kick and routing stats")
    else:
        print("Using standard callbacks for dense model")
    
    return callbacks

def execute_training(trainer, resume_from_checkpoint=None):
    """Execute the training loop."""
    logger.info("Starting training...")
    
    try:
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            logger.info("Starting fresh training")
            trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        trainer.save_model()
        logger.info("Model saved before exit")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Try to save current state
        try:
            trainer.save_model()
            logger.info("Model saved after training error")
        except Exception as save_error:
            logger.error(f"Failed to save model after error: {save_error}")
        raise

def save_final_model(trainer, output_dir):
    """Save the final trained model."""
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    return final_model_path
