import os
import torch
import logging
import argparse
import time
import numpy as np
import glob
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    GPT2Config, GPT2Tokenizer, AutoTokenizer, Trainer, TrainingArguments,
    default_data_collator, set_seed, DataCollatorForLanguageModeling
)
from bitsandbytes.optim import Adam8bit
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
from datetime import datetime
import pathlib
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import dataset loaders from the correct module path
from data_modules import (
    create_openassistant_loader,
    create_anthropic_hh_loader,
    create_sharegpt_loader,
    create_alpaca_loader,
    create_wikipedia_loader,
    create_the_pile_loader,
    create_conversational_mix,
    DatasetMixer
)

# Import the actual Dataset class from datasets
from datasets import Dataset

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
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = None
        self.last_custom_loss = None  # Store our custom loss value

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar = tqdm(total=self.total_steps, desc="[Training]", unit="step", dynamic_ncols=True)
        logger.info(f"🚀 Starting training for {self.total_steps} steps")
        logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {args.per_device_train_batch_size * args.world_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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

class LrKickOnceCallback(TrainerCallback):
    """Multiply learning rate by a factor once at a specific step to break plateaus."""
    def __init__(self, factor=3.0, at_step=0):
        self.factor = factor
        self.at_step = at_step
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.done and state.global_step >= self.at_step:
            opt = kwargs.get("optimizer")
            sched = kwargs.get("lr_scheduler") 
            if opt and sched:
                # Boost current LR
                for g in opt.param_groups:
                    g["lr"] *= self.factor
                # Update scheduler base LRs so it doesn't reset
                if hasattr(sched, 'base_lrs'):
                    sched.base_lrs = [lr * self.factor for lr in sched.base_lrs]
                logger.info(f"🚀 LR kicked by ×{self.factor} at step {state.global_step}")
                self.done = True

class MoERoutingCallback(TrainerCallback):
    """Log MoE expert utilization statistics."""
    def __init__(self, log_every=500):
        self.log_every = log_every
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0:
            logger.info(f"🔍 MoE Debug @ Step {state.global_step}: Callback triggered")
            model = kwargs.get("model")
            if model and hasattr(model, 'transformer'):
                self._log_routing_stats(model, state.global_step)
    
    def _log_routing_stats(self, model, step):
        """Extract and log expert routing statistics."""
        try:
            expert_counts = []
            total_tokens = 0
            aux_losses = []
            
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # Check each MoE block
                for idx, block in enumerate(model.transformer.h):
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'moe'):
                        moe_layer = block.mlp.moe
                        
                        # Get routing statistics from Tutel MOELayer
                        if hasattr(moe_layer, 'dispatch_count') and moe_layer.dispatch_count is not None:
                            counts = moe_layer.dispatch_count.cpu().numpy()
                            expert_counts.append(counts)
                            total_tokens += counts.sum()
                        
                        # Get auxiliary loss (load balancing)
                        if hasattr(moe_layer, 'l_aux') and moe_layer.l_aux is not None:
                            aux_loss_val = moe_layer.l_aux if isinstance(moe_layer.l_aux, float) else moe_layer.l_aux.item()
                            aux_losses.append(aux_loss_val)
            
            if expert_counts:
                # Calculate utilization stats
                all_counts = torch.cat([torch.tensor(c, dtype=torch.float32) for c in expert_counts])
                total_routed = all_counts.sum().item()
                expert_util = (all_counts > 0).float().mean().item() * 100  # % of experts used
                load_balance = all_counts.std().item() / (all_counts.mean().item() + 1e-8)  # CV
                
                # Calculate per-layer stats
                layer_stats = []
                for i, counts in enumerate(expert_counts):
                    layer_util = (counts > 0).sum() / len(counts) * 100
                    layer_cv = counts.std() / (counts.mean() + 1e-8)
                    max_load = counts.max()
                    layer_stats.append(f"L{i*2}:{layer_util:.0f}%/{layer_cv:.2f}/{max_load}")
                
                avg_aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
                
                logger.info(f"🧠 MoE Routing @ Step {step}: "
                          f"Overall Util: {expert_util:.1f}%, CV: {load_balance:.3f}, "
                          f"Aux Loss: {avg_aux_loss:.3f}, Total: {total_routed}")
                logger.info(f"📊 Per-Layer [Util%/CV/MaxLoad]: {' | '.join(layer_stats)}")
            else:
                logger.info(f"🔍 MoE Debug: No expert counts found - expert_counts is empty")
        except Exception as e:
            logger.error(f"MoE routing stats failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
def _get_size_mb(ds):
    """Roughly estimate dataset size in MB.

    1. Use the `dataset_size` provided in the dataset `info` dict when available.
    2. If the original `text` column is still present, sum the UTF-8 byte lengths of
       the examples.
    3. Otherwise (e.g. after tokenisation where only `input_ids` remain) estimate by
       counting tokens: `len(input_ids) * 2` bytes (2-byte per token rough guess).
    """
    size_bytes = None

    # 1) Try metadata if it exists
    if hasattr(ds, "info"):
        size_bytes = getattr(ds.info, "dataset_size", None) or getattr(ds.info, "size_in_bytes", None)

    # 2) Fallback: compute from columns
    if size_bytes is None:
        try:
            if "text" in ds.column_names:
                # Sum UTF-8 bytes of text
                size_bytes = sum(len(t.encode("utf-8")) for t in ds["text"])
            elif "input_ids" in ds.column_names:
                # Rough estimate: 2 bytes per token (uint16/INT)
                size_bytes = sum(len(ids) for ids in ds["input_ids"]) * 2
            else:
                size_bytes = 0
        except Exception:
            size_bytes = 0

    return (size_bytes or 0) / (1024 * 1024)


def _print_split_stats(name, train_ds, val_ds):
    print(f"{name} - Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"{name} sizes -> Train: {_get_size_mb(train_ds):.1f} MB , Val: {_get_size_mb(val_ds):.1f} MB , Total: {_get_size_mb(train_ds)+_get_size_mb(val_ds):.1f} MB")


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_openassistant_dataset(tok, max_len=256):
    """Load and preprocess the OpenAssistant OASST1 dataset."""
    from datasets import load_dataset, DatasetDict, concatenate_datasets
    
    print("Loading OpenAssistant OASST1 dataset from HuggingFace...")
    ds = load_dataset("OpenAssistant/oasst1")
    
    print("Processing dataset...")
    # If the dataset has multiple splits, concatenate them
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
    
    print("Filtering for English assistant responses...")
    # Filter for English assistant responses
    ds = ds.filter(lambda x: x.get("lang") == "en" and x.get("role") == "assistant")
    
    print("Splitting into train/validation...")
    # Split into train and validation
    split = ds.train_test_split(test_size=0.05, seed=42)
    
    def tok_fn(batch):
        # Tokenize without padding; dynamic padding will be added by the data collator
        return tok(batch["text"], truncation=True, max_length=max_len)
    
    print("Tokenizing training set...")
    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=split["train"].column_names)
    
    print("Tokenizing validation set...")
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=split["test"].column_names)
    
    # Remove examples that became empty after tokenization
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 1)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) > 1)
    
    _print_split_stats("OpenAssistant", train_ds, val_ds)
    return train_ds, val_ds


def load_pile_dataset(tokenizer, max_len=256, max_samples=100000):
    """Load and preprocess a subset of The Pile dataset."""
    from datasets import load_dataset, concatenate_datasets
    
    print("Loading The Pile dataset from HuggingFace...")
    
    # Load a subset of The Pile
    datasets = []
    subsets = ["enron_emails", "pubmed"]  # Using smaller subsets that fit in ~1GB
    
    for subset in subsets:
        try:
            print(f"Loading {subset} subset...")
            ds = load_dataset("EleutherAI/pile", subset, split=f"train[:{max_samples//len(subsets)}]")
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {subset}: {str(e)}")
    
    if not datasets:
        raise ValueError("Failed to load any Pile subsets")
        
    # Combine all subsets
    combined = concatenate_datasets(datasets)
    
    # Filter out very short or long examples
    combined = combined.filter(lambda x: 50 < len(x["text"].split()) < 1000)
    
    # Split into train and validation
    split = combined.train_test_split(test_size=0.05, seed=42)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding="max_length"
        )
    
    print("Tokenizing training set...")
    train_ds = split["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["train"].column_names,
        num_proc=4
    )
    
    print("Tokenizing validation set...")
    val_ds = split["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["test"].column_names,
        num_proc=4
     )
     
    _print_split_stats("The Pile", train_ds, val_ds)
    return train_ds, val_ds


def load_wikipedia_dataset(tokenizer, max_len=256, language="20220301.en", percent=1):
    """Load and preprocess a slice of the Wikipedia dump.
    `percent` specifies the percentage slice of the full dump to keep (1 ≈ ~250 MB for English).
    """
    from datasets import load_dataset

    slice_str = f"train[:{percent}%]" if percent < 100 else "train"
    print(f"Loading Wikipedia ({language}) dataset slice {slice_str} …")

    ds = load_dataset("wikipedia", language, split=slice_str)

    # Filter very short articles
    ds = ds.filter(lambda x: len(x["text"].split()) > 50)

    # Train/val split
    split = ds.train_test_split(test_size=0.05, seed=42)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    print("Tokenizing Wikipedia training set…")
    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=split["train"].column_names,
                                   num_proc=4)

    print("Tokenizing Wikipedia validation set…")
    val_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=split["test"].column_names,
                                  num_proc=4)
 
    _print_split_stats("Wikipedia", train_ds, val_ds)
    return train_ds, val_ds


def load_conversational_dataset(tokenizer, max_len=256, max_samples=50000):
    """
    Load and preprocess the Anthropic HH-RLHF dataset for conversational training.
    This dataset contains high-quality human-AI conversations that will improve
    the model's conversational abilities.
    """
    from datasets import load_dataset
    
    print("Loading Anthropic HH-RLHF conversational dataset...")
    
    # Load the helpful subset of the HH-RLHF dataset
    try:
        print(f"Loading helpful-base subset...")
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        ds = ds["train"]  # Use training split
        
        # Take a subset if specified
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            
        print(f"Loaded {len(ds)} conversational examples")
        
        def format_conversation(example):
            """Format the conversation into a single text string"""
            chosen = example["chosen"]
            # Clean up the conversation format
            text = chosen.strip()
            # Ensure it ends with an EOS token
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            return {"text": text}
        
        # Format conversations
        ds = ds.map(format_conversation, remove_columns=ds.column_names)
        
        # Filter out very short or very long conversations
        ds = ds.filter(lambda x: 50 <= len(x["text"].split()) <= 400)
        
        print(f"After filtering: {len(ds)} conversational examples")
        
        # Split into train/validation
        split = ds.train_test_split(test_size=0.05, seed=42)
        
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_len,
                return_tensors=None
            )
        
        # Tokenize
        train_ds = split["train"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["train"].column_names,
            num_proc=4
        )
        val_ds = split["test"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=split["test"].column_names,
            num_proc=4
        )
        
        print(f"Conversational dataset - Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds
        
    except Exception as e:
        print(f"Failed to load Anthropic HH-RLHF dataset: {e}")
        print("Falling back to alternative conversational dataset...")
        
        # Fallback to a simpler conversational dataset
        try:
            # Use a subset of the original OpenAssistant but format differently for conversations
            ds = load_dataset("OpenAssistant/oasst1")
            from datasets import concatenate_datasets, DatasetDict
            if isinstance(ds, DatasetDict):
                ds = concatenate_datasets([ds[k] for k in ds.keys()])
            
            # Filter for English conversations (both human and assistant)
            ds = ds.filter(lambda x: x["lang"] == "en" and len(x["text"].split()) > 20)
            
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            
            def format_oasst_conversation(example):
                text = example["text"].strip()
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                return {"text": text}
            
            ds = ds.map(format_oasst_conversation, remove_columns=ds.column_names)
            split = ds.train_test_split(test_size=0.05, seed=42)
            
            def tokenize_fn(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=max_len,
                    return_tensors=None
                )
            
            train_ds = split["train"].map(
                tokenize_fn, 
                batched=True, 
                remove_columns=split["train"].column_names,
                num_proc=4
            )
            val_ds = split["test"].map(
                tokenize_fn, 
                batched=True, 
                remove_columns=split["test"].column_names,
                num_proc=4
            )
            
            print(f"Fallback conversational dataset - Train: {len(train_ds)}, Val: {len(val_ds)}")
            return train_ds, val_ds
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("No conversational dataset loaded")
            return None, None


def load_datasets_with_loader(tokenizer, args):
    """Load datasets using the new modular system."""
    datasets = []
    
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
        
        # If no datasets were loaded, use OpenAssistant as default
        if not datasets:
            oa_loader = create_openassistant_loader(
                max_samples=args.max_samples,
                max_length=args.max_length
            )
            oa_train, oa_val, _ = oa_loader.load_and_process()
            datasets.append(("openassistant", oa_train, oa_val))
        
        # Combine all datasets
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
        
    except Exception as e:
        logger.error(f"Error in dataset loading: {e}")
        raise

def combine_datasets(dataset1, dataset2):
    """Combine two datasets by interleaving their examples."""
    combined_train = concatenate_datasets([dataset1[0], dataset2[0]])
    combined_val = concatenate_datasets([dataset1[1], dataset2[1]])
    
    # Shuffle the combined datasets
    combined_train = combined_train.shuffle(seed=42)
    combined_val = combined_val.shuffle(seed=42)
    
    return combined_train, combined_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_pile", action="store_true", help="Use The Pile dataset in addition to OpenAssistant")
    parser.add_argument("--use_wiki", action="store_true", help="Use Wikipedia dataset in addition to OpenAssistant")
    parser.add_argument("--use_conversation", action="store_true", help="Use conversational dataset (HH-RLHF) to improve chat abilities")
    parser.add_argument("--use_sharegpt", action="store_true", help="Use ShareGPT dataset for training")
    parser.add_argument("--use_alpaca", action="store_true", help="Use Alpaca instruction dataset for training")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples per individual dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--ignore_checkpoint", action="store_true", help="Ignore existing checkpoints and start training from step 1")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio of total steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
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

    # Load datasets using the new modular system
    print("Loading datasets using the modular dataset system...")
    try:
        train_ds, val_ds = load_datasets_with_loader(tokenizer, args)
        
        # Convert to Dataset objects if they're lists
        if isinstance(train_ds, list):
            train_ds = Dataset.from_dict({"text": [item["text"] for item in train_ds]})
            val_ds = Dataset.from_dict({"text": [item["text"] for item in val_ds]})
            
            # Tokenize the datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt"
                )
                
            train_ds = train_ds.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing training data"
            )
            val_ds = val_ds.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing validation data"
            )
            
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        logger.info("Falling back to default dataset loading...")
        
        # Fallback to original dataset loading
        print("Loading OpenAssistant OASST1 dataset...")
        oa_train_ds, oa_val_ds = load_openassistant_dataset(tokenizer, max_len=args.max_length)
        
        # Optionally add conversational dataset first
        if args.use_conversation:
            print("Loading conversational dataset for improved chat abilities...")
            conv_train_ds, conv_val_ds = load_conversational_dataset(tokenizer, max_len=args.max_length, max_samples=30000)
            if conv_train_ds is not None and conv_val_ds is not None:
                oa_train_ds, oa_val_ds = combine_datasets((oa_train_ds, oa_val_ds), (conv_train_ds, conv_val_ds))
                print("Successfully integrated conversational dataset!")
            else:
                print("Failed to load conversational dataset, continuing with existing datasets...")
        
        # Optionally add Wikipedia
        if args.use_wiki:
            print("Loading Wikipedia dataset...")
            wiki_train_ds, wiki_val_ds = load_wikipedia_dataset(tokenizer, max_len=args.max_length)
            oa_train_ds, oa_val_ds = combine_datasets((oa_train_ds, oa_val_ds), (wiki_train_ds, wiki_val_ds))

        if args.use_pile:
            print("Loading The Pile dataset...")
            pile_train_ds, pile_val_ds = load_pile_dataset(tokenizer, max_len=args.max_length)
            
            print("Combining datasets...")
            train_ds, val_ds = combine_datasets(
                (oa_train_ds, oa_val_ds),
                (pile_train_ds, pile_val_ds)
            )
        else:
            train_ds, val_ds = oa_train_ds, oa_val_ds
    
    _print_split_stats("Final Combined", train_ds, val_ds)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Build dataset description
    dataset_parts = ["OpenAssistant"]
    if args.use_conversation:
        dataset_parts.append("Conversational (HH-RLHF)")
    if args.use_wiki:
        dataset_parts.append("Wikipedia")
    if args.use_pile:
        dataset_parts.append("The Pile")
    
    dataset_description = " + ".join(dataset_parts)
    print(f"Using {dataset_description} dataset")
    
    # Tokenization is already handled in the dataset loading functions

    # Model configuration - slightly larger model
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,  # Increased context length
        n_embd=1024,  # Increased model capacity
        n_layer=8,  # Increased from 6
        n_head=16,  # Increased from 12
        n_inner=4096,  # Increased feed-forward dimension
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

    # Create a custom trainer with safetensors saving
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Compute loss including MoE auxiliary loss for load balancing.
            """
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**inputs)
            
            # Get the main language modeling loss
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                lm_loss = outputs.loss
            
            # Collect auxiliary losses from all MoE layers
            aux_loss = 0.0
            aux_loss_weight = 0.05  # Increased from 0.01 to 0.05 for stronger load balancing
            
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                for block in model.transformer.h:
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'moe'):
                        moe_layer = block.mlp.moe
                        
                        # Get routing statistics from Tutel MOELayer
                        if hasattr(moe_layer, 'l_aux') and moe_layer.l_aux is not None:
                            if isinstance(moe_layer.l_aux, torch.Tensor):
                                aux_loss += moe_layer.l_aux
                            else:
                                aux_loss += float(moe_layer.l_aux)
            
            # Total loss = language modeling loss + weighted auxiliary loss
            total_loss = lm_loss + aux_loss_weight * aux_loss
            
            # Store the last computed loss for logging callback
            self._last_total_loss = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
            
            # Log the loss components occasionally
            if hasattr(self, '_loss_log_counter'):
                self._loss_log_counter += 1
            else:
                self._loss_log_counter = 1
                
            if self._loss_log_counter % 100 == 0:  # Log every 100 steps
                logger.info(f"Loss breakdown: LM={lm_loss:.4f}, Aux={aux_loss:.4f}, Total={total_loss:.4f}")
            
            return (total_loss, outputs) if return_outputs else total_loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            """
            Override training step to ensure our custom loss is used and logged correctly.
            """
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            # Store the loss for logging - this is the key fix
            self.state.log_history.append({"loss": loss.item(), "step": self.state.global_step})
            
            return loss.detach() / self.args.gradient_accumulation_steps

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

        def _load_optimizer_and_scheduler(self, checkpoint):
            """Override to skip loading optimizer/scheduler state to avoid PyTorch security check."""
            logger.info("Skipping optimizer/scheduler state loading to avoid PyTorch security check")
            logger.info("Starting with fresh optimizer/scheduler state")
            pass

        def _load_scaler(self, checkpoint):
            """Override to skip loading scaler state to avoid PyTorch security check."""
            logger.info("Skipping scaler state loading to avoid PyTorch security check")
            pass

        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
            """Create a CosineAnnealingWarmRestarts LR scheduler.

            The first restart happens after ~20 % of the total remaining steps (T_0) and
            each subsequent cycle is twice as long (T_mult=2). This gives smooth
            decay punctuated by periodic warm restarts which can help escape loss
            plateaus without manual intervention.
            """
            if self.lr_scheduler is None:
                optimizer = optimizer or self.optimizer
                t0 = max(1, num_training_steps // 5)
                logger.info(f"Building Cosine LR scheduler: T0={t0}, T_mult=2, steps={num_training_steps}")
                self.lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=t0,
                    T_mult=2,
                )
            return self.lr_scheduler

    # Training arguments with enhanced settings (compatible with older Transformers)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        # Checkpointing
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep last 3 checkpoints
        # Performance
        fp16=True,
        dataloader_num_workers=4,  # Use more workers for faster data loading
        disable_tqdm=False,  # Show progress bar
        remove_unused_columns=True,
        max_grad_norm=1.0,  # Gradient clipping
        local_rank=-1,
        no_cuda=False
    )

    # Use standard data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling
    )

    # Initialize callbacks
    progress_callback = ProgressCallback(training_args.max_steps)
    lr_kick_callback = LrKickOnceCallback(factor=5.0, at_step=44500)  # Re-enabled to break plateau
    moe_routing_callback = MoERoutingCallback(log_every=5)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Find the latest checkpoint if it exists
    latest_checkpoint = None
    resume_from_checkpoint = None
    if os.path.isdir(args.output_dir) and not args.ignore_checkpoint:
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            # Sort by step number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            resume_from_checkpoint = latest_checkpoint
            print(f"Found checkpoint: {latest_checkpoint}")
    elif os.path.isdir(args.output_dir) and args.ignore_checkpoint:
        # Find latest checkpoint for model weights but don't resume training from it
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            print(f"Found checkpoint for model weights: {latest_checkpoint}")
            print("Will load model weights but start training from step 1 (--ignore_checkpoint)")
        else:
            print("No checkpoints found, starting fresh training")
    
    # Load model weights from checkpoint if exists
    if latest_checkpoint and os.path.isdir(latest_checkpoint) and not args.ignore_checkpoint:
        print(f"Loading model weights from {latest_checkpoint}")
        # Load the model state dict directly
        model_path = os.path.join(latest_checkpoint, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully")
        else:
            print(f"Warning: Model weights not found at {model_path}")
    elif latest_checkpoint and os.path.isdir(latest_checkpoint) and args.ignore_checkpoint:
        print(f"Loading model weights from {latest_checkpoint} but starting from step 1")
        # Load the model state dict directly
        model_path = os.path.join(latest_checkpoint, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully")
        else:
            print(f"Warning: Model weights not found at {model_path}")
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[
            progress_callback, 
            #lr_kick_callback, 
            moe_routing_callback, 
            CheckpointInfoCallback()
        ],
    )
    
    # Start fresh training (with loaded weights)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    main()
