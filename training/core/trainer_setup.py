"""
Training setup and custom trainer implementation.
"""
import os
import logging
import torch
import safetensors.torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Get logger
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """Custom trainer with safetensors saving and MoE auxiliary loss support."""
    
    def __init__(self, *args, architecture="moe", **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss including MoE auxiliary loss for load balancing (MoE only).
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        
        if self.architecture == "moe":
            # MoE models include auxiliary loss for load balancing
            loss = outputs.loss
            if hasattr(outputs, 'auxiliary_loss') and outputs.auxiliary_loss is not None:
                # Add auxiliary loss for load balancing (usually small weight)
                aux_loss_weight = 0.01  # Small weight for auxiliary loss
                total_loss = loss + aux_loss_weight * outputs.auxiliary_loss
                
                # Log the losses separately
                self.log({
                    "train/main_loss": loss.item(),
                    "train/aux_loss": outputs.auxiliary_loss.item(),
                    "train/total_loss": total_loss.item()
                })
                
                return (total_loss, outputs) if return_outputs else total_loss
            else:
                return (loss, outputs) if return_outputs else loss
        else:
            # Dense models use standard language modeling loss
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model using safetensors format."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Save the model state dict using safetensors
        model_path = os.path.join(output_dir, "model.safetensors")
        safetensors.torch.save_file(self.model.state_dict(), model_path)
        
        # Save the tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save the training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        logger.info(f"Model saved to {output_dir}")

def setup_training_args(args, output_dir):
    """Setup training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_steps=500,
        evaluation_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=f"{args.architecture}_model_{args.model_size if hasattr(args, 'model_size') else 'moe'}",
        seed=42,
        data_seed=42,
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        prediction_loss_only=True,
    )
    
    return training_args

def setup_data_collator(tokenizer):
    """Setup data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,
    )

def create_trainer(model, training_args, train_dataset, val_dataset, tokenizer, callbacks, architecture="moe"):
    """Create custom trainer with callbacks."""
    data_collator = setup_data_collator(tokenizer)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        architecture=architecture,
    )
    
    return trainer
