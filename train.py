import os
import torch
import logging
import argparse
import numpy as np
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config, GPT2Tokenizer, AutoTokenizer, Trainer, TrainingArguments,
    default_data_collator, set_seed
)
from transformers import DataCollatorForLanguageModeling
from bitsandbytes.optim import Adam8bit
import logging
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
from datetime import datetime
import pathlib

# Insert a constant MODEL_SCALE (0.8) after the imports so that model size is reduced by 20%
MODEL_SCALE = 1.0

# Ensure tokenizers parallelism disabled
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

class _SuppressCapacityLogs(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return 'Capacity =' not in msg and 'real-time capacity-factor' not in msg

logging.getLogger().addFilter(_SuppressCapacityLogs())
logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.total_steps = None
        self.current_step = 0
        self.start_time = None
        self.last_log_time = 0
        self.log_interval = 10  # Log every 10 steps

    def format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        if seconds <= 0:
            return "--:--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import time
        self.total_steps = state.max_steps
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Initialize progress bar with more detailed format
        self.pbar = tqdm(
            total=self.total_steps,
            desc="[Training]",
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        logger.info(f"🚀 Starting training for {self.total_steps} steps")
        if args.local_rank in [-1, 0]:
            logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size = {args.per_device_train_batch_size * args.world_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import time
        current_time = time.time()
        self.current_step = state.global_step
        
        # Calculate metrics
        elapsed = current_time - self.start_time
        steps_done = self.current_step
        steps_left = self.total_steps - steps_done
        
        # Calculate ETA
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
        eta = steps_left / steps_per_sec if steps_per_sec > 0 else 0
        
        # Update progress bar
        self.pbar.update(1)
        self.pbar.set_postfix({
            'loss': f"{state.log_history[-1].get('loss', 0):.4f}" if state.log_history else 'N/A',
            'lr': f"{state.log_history[-1].get('learning_rate', 0):.2e}" if state.log_history else 'N/A',
            'ETA': self.format_time(eta)
        })
        
        # Log detailed progress periodically
        if (current_time - self.last_log_time > 30 or  # Every 30 seconds
            steps_done % 100 == 0 or 
            steps_done == self.total_steps):
            
            progress_pct = (steps_done / self.total_steps) * 100
            elapsed_str = self.format_time(elapsed)
            eta_str = self.format_time(eta)
            
            logger.info(
                f"📊 Step {steps_done}/{self.total_steps} "
                f"({progress_pct:.1f}%) | "
                f"Loss: {state.log_history[-1].get('loss', 0):.4f} | "
                f"LR: {state.log_history[-1].get('learning_rate', 0):.2e} | "
                f"Elapsed: {elapsed_str} | "
                f"ETA: {eta_str}"
            )
            self.last_log_time = current_time
            
        # Final log at the end of training
        if steps_done == self.total_steps:
            total_time = self.format_time(elapsed)
            logger.info(f"\n✨ Training completed in {total_time} ✨")
            self.pbar.close()
            logger.info(f"Step {self.current_step}/{self.total_steps} - ETA: {eta_str}")
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                logger.info(f"Step {self.current_step}/{self.total_steps} - Loss: {last_log['loss']}")
            if 'eval_loss' in last_log:
                logger.info(f"Step {self.current_step}/{self.total_steps} - Eval Loss: {last_log['eval_loss']}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            logger.info(f"Evaluation at step {state.global_step}: {metrics}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar.close()
        logger.info("Training completed.")

class CheckpointInfoCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.best_eval_loss = None
        self.start_time = None
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def on_save(self, args, state, control, **kwargs):
        import time
        output_dir = args.output_dir
        checkpoints = sorted(
            glob.glob(os.path.join(output_dir, 'checkpoint-*')),
            key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else -1
        )
        if not checkpoints:
            return
        latest_ckpt = checkpoints[-1]
        # Gather info
        step = state.global_step
        epoch = getattr(state, 'epoch', None)
        train_loss = state.log_history[-1]['loss'] if state.log_history and 'loss' in state.log_history[-1] else None
        eval_loss = state.log_history[-1]['eval_loss'] if state.log_history and 'eval_loss' in state.log_history[-1] else None
        if eval_loss is not None:
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
        percent_complete = float(step) / float(state.max_steps) * 100 if state.max_steps else None
        samples_seen = step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        # Learning rate (from optimizer)
        lr = None
        if hasattr(kwargs.get('optimizer', None), 'param_groups'):
            lr = kwargs['optimizer'].param_groups[0]['lr']
        # Time per step
        now = time.time()
        if self.last_step_time:
            time_per_step = (now - self.last_step_time)
        else:
            time_per_step = None
        self.last_step_time = now
        # GPU memory usage (if available)
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # Model config
        model_config = getattr(kwargs.get('model', None), 'config', None)
        param_count = None
        if model_config and hasattr(model_config, 'to_dict'):
            param_count = sum(p.numel() for p in kwargs['model'].parameters()) if kwargs.get('model', None) else None
        # Markdown content
        md = f"""
# Checkpoint Info

- **Checkpoint Directory:** `{latest_ckpt}`
- **Step:** {step}
- **Epoch:** {epoch if epoch is not None else 'N/A'}
- **Timestamp:** {datetime.now().isoformat()}
- **Samples Seen:** {samples_seen}
- **Percent Complete:** {percent_complete:.2f}%
- **Learning Rate:** {lr if lr is not None else 'N/A'}
- **Train Loss:** {train_loss if train_loss is not None else 'N/A'}
- **Eval Loss:** {eval_loss if eval_loss is not None else 'N/A'}
- **Best Eval Loss So Far:** {self.best_eval_loss if self.best_eval_loss is not None else 'N/A'}
- **Time per Step:** {time_per_step:.3f} sec
- **GPU Memory Usage:** {gpu_mem:.2f} GB if gpu_mem is not None else 'N/A'
- **Parameter Count:** {param_count if param_count is not None else 'N/A'}

## Training Arguments
```
{json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)}
```

## Model Config
```
{model_config.to_dict() if model_config and hasattr(model_config, 'to_dict') else model_config}
```
"""
        info_path = os.path.join(latest_ckpt, 'checkpoint-info.md')
        with open(info_path, 'w') as f:
            f.write(md)

def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(tok, max_len=1024):
    ds = load_dataset("OpenAssistant/oasst1")
    # If ds is a DatasetDict, concatenate all splits
    if isinstance(ds, dict):
        from datasets import concatenate_datasets
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
    # Now filter
    ds = ds.filter(lambda x: x["lang"] == "en" and x["role"] == "assistant")
    # Now split
    split = ds.train_test_split(test_size=0.05, seed=42)
    def tok_fn(batch):
        # Tokenise without padding; dynamic padding will be added by the data collator.
        # This prevents sequences made entirely of pad tokens which caused zero loss.
        return tok(batch["text"], truncation=True, max_length=max_len)
    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=split["train"].column_names)
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=split["test"].column_names)
    # Remove examples that became empty after tokenisation
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 1)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) > 1)
    return train_ds, val_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a simple text dataset for testing
    print("Loading wikitext dataset for testing...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Filter out empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Take a small subset for training and validation
    train_ds = dataset["train"].select(range(100))  # Small subset for testing
    val_ds = dataset["validation"].select(range(20))  # Small validation set

    def tokenize_function(examples):
        # Simple tokenization with padding and truncation
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
    # Apply tokenization
    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # Model configuration - reduced size for 16GB GPU
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,  # Reduced context length
        n_embd=768,  # Reduced from 1024
        n_layer=6,  # Reduced from 12
        n_head=12,  # Reduced from 16
        n_inner=3072,  # Reduced from 4096
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    from moe_model import GPT2WithMoE
    model = GPT2WithMoE(cfg)

    # Training arguments with basic checkpointing
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        # Logging
        logging_steps=10,  # Log every 10 steps
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        # Checkpointing
        save_steps=100,  # Save checkpoint every 100 steps
        save_total_limit=3,  # Keep only 3 checkpoints
        # Performance
        fp16=True,
        dataloader_num_workers=0,
        disable_tqdm=True,  # We use our own progress bar
        remove_unused_columns=True,
        max_grad_norm=1.0,  # Gradient clipping
        local_rank=-1,
        no_cuda=False,
    )

    # Use standard data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling
    )

    # Initialize callbacks
    progress_callback = ProgressCallback()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Create a custom trainer with safetensors saving
    class CustomTrainer(Trainer):
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a new state dict without shared tensors
            state_dict = {}
            for name, param in self.model.named_parameters():
                if param is not None:
                    state_dict[name] = param.detach().clone()
            
            # Save using safetensors
            model_path = os.path.join(output_dir, "model.safetensors")
            from safetensors.torch import save_file
            save_file(state_dict, model_path)
            
            # Also save using PyTorch's native format as a fallback
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            
            # Save config
            self.model.config.save_pretrained(output_dir)
            
            # Save tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model saved to {output_dir}")
            
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # Custom load to handle safetensors
            from safetensors.torch import load_file
            import os
            
            model = self.model if model is None else model
            
            # Try loading from safetensors first
            safetensors_path = os.path.join(resume_from_checkpoint, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
                return model
                
            # Fall back to default loading
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
    
    # Initialize trainer with custom saving
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[progress_callback],
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
