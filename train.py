import argparse, random, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, Trainer, TrainingArguments
from bitsandbytes.optim import Adam8bit
from moe_model import GPT2WithMoE
import logging
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

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

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.total_steps = state.max_steps
        self.pbar = tqdm(total=self.total_steps, desc="Training Progress", dynamic_ncols=True)
        logger.info(f"Training started: {self.total_steps} steps expected.")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_step = state.global_step
        self.pbar.update(1)
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


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(tok, max_len=1024):
    ds = load_dataset("OpenAssistant/oasst1")
    ds = ds.filter(lambda x: x["lang"] == "en" and x["role"] == "assistant")
    split = ds.train_test_split(test_size=0.05, seed=42)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
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
        evaluation_strategy="steps",
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
        callbacks=[ProgressCallback()]
    )
    trainer.train()

if __name__ == "__main__":
    main()
