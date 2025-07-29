"""
Progress tracking callback for training.
"""
import time
import logging
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# Get logger
logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    """Callback to track training progress with progress bar and ETA."""
    
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = None
        self.last_custom_loss = None  # Store our custom loss value

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize progress bar and log training start."""
        self.pbar = tqdm(total=self.total_steps, desc="[Training]", unit="step", dynamic_ncols=True)
        logger.info(f"🚀 Starting training for {self.total_steps} steps")
        logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {args.per_device_train_batch_size * args.world_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Update progress bar and log step information."""
        self.current_step = state.global_step
        
        # Get the custom loss from the trainer if available
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, '_loss_log_counter') and hasattr(trainer, '_last_total_loss'):
            loss_to_display = trainer._last_total_loss
        elif state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            loss_to_display = last_log.get('loss', 'N/A')
        else:
            loss_to_display = 'N/A'
            
        # Update progress bar
        if self.pbar:
            lr = state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * (self.pbar.format_dict.get('elapsed', 0) / max(self.current_step, 1))
            eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
            
            self.pbar.set_postfix({
                'loss': f'{loss_to_display:.4f}' if isinstance(loss_to_display, (int, float)) else str(loss_to_display),
                'lr': f'{lr:.2e}',
                'ETA': eta_str
            })
            self.pbar.update(1)

        # Calculate and display ETA
        if self.current_step > 0:
            elapsed_time = time.time() - (state.log_history[0].get('timestamp', time.time()) if state.log_history else time.time())
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "calculating..."

        if self.current_step == self.total_steps:
            total_time = time.time() - (state.log_history[0].get('timestamp', time.time()) if state.log_history else time.time())
            total_time_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
            logger.info(f"\n✨ Training completed in {total_time_str} ✨")
            self.pbar.close()
            logger.info(f"Step {self.current_step}/{self.total_steps} - ETA: {eta_str}")
        
        # Log the correct loss value
        if isinstance(loss_to_display, (int, float)):
            logger.info(f"Step {self.current_step}/{self.total_steps} - Loss: {loss_to_display:.4f}")
        else:
            logger.info(f"Step {self.current_step}/{self.total_steps} - Loss: {loss_to_display}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            logger.info(f"Evaluation at step {state.global_step}: {metrics}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Clean up progress bar on training end."""
        self.pbar.close()
