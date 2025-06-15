import argparse, random, os, torch, glob
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, Trainer, TrainingArguments
from bitsandbytes.optim import Adam8bit
from moe_model import GPT2WithMoE
import logging
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.total_steps = None
        self.current_step = 0
        self.start_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import time
        self.total_steps = state.max_steps
        self.pbar = tqdm(total=self.total_steps, desc="Training Progress", dynamic_ncols=True)
        self.start_time = time.time()
        logger.info(f"Training started: {self.total_steps} steps expected.")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import time
        self.current_step = state.global_step
        self.pbar.update(1)
        elapsed = time.time() - self.start_time if self.start_time else 0
        steps_done = self.current_step
        steps_left = self.total_steps - steps_done
        eta = (elapsed / steps_done * steps_left) if steps_done > 0 else 0
        # Format ETA as H:MM:SS
        def format_eta(seconds):
            if seconds <= 0:
                return "--:--:--"
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
        eta_str = format_eta(eta)
        self.pbar.set_postfix({"ETA": eta_str})
        if steps_done % 100 == 0 or steps_done == self.total_steps:
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
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=split["train"].column_names)
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=split["test"].column_names)
    return train_ds, val_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()

    seed_all()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_data(tokenizer)

    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=1280,
        n_layer=24,
        n_head=20,
        n_inner=5120,
        pad_token_id=tokenizer.pad_token_id
    )

    model = GPT2WithMoE(cfg)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        save_total_limit=2,
        deepspeed="./ds_config.json" if args.use_deepspeed else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[ProgressCallback(), CheckpointInfoCallback()]
    )

    # Find latest checkpoint if exists
    checkpoint_dir = args.output_dir
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else -1)
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()
