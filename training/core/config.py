"""
Training configuration and argument parsing.
"""
import argparse
import os

# Model scale constant
MODEL_SCALE = 1.0

def setup_environment():
    """Set up environment variables."""
    # Ensure tokenizers parallelism disabled
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_arguments():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train transformer models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=1)
    
    # Architecture selection
    parser.add_argument("--architecture", type=str, choices=["moe", "dense"], default="moe", 
                       help="Choose model architecture: 'moe' for MoE transformer, 'dense' for standard transformer")
    parser.add_argument("--model_size", type=str, choices=["medium", "large", "200m", "400m", "600m", "800m", "1.5b"], default="800m",
                       help="Dense model size (ignored for MoE architecture)")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Use flash attention for memory efficiency (dense models only)")
    parser.add_argument("--no_flash_attention", action="store_true", 
                       help="Disable flash attention (use standard attention)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Enable mixed precision training with FP16 (default: True)")
    parser.add_argument("--no_fp16", action="store_true",
                       help="Disable FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 instead of FP16 (requires newer hardware)")
    
    # Dataset selection
    parser.add_argument("--use_pile", action="store_true", help="Use The Pile dataset in addition to OpenAssistant")
    parser.add_argument("--use_wiki", action="store_true", help="Use Wikipedia dataset in addition to OpenAssistant")
    parser.add_argument("--use_conversation", action="store_true", default=True, help="Use conversational dataset (HH-RLHF) to improve chat abilities")
    parser.add_argument("--use_sharegpt", action="store_true", help="Use ShareGPT dataset for training")
    parser.add_argument("--use_alpaca", action="store_true", default=True, help="Use Alpaca instruction dataset for training")
    parser.add_argument("--use_ultrachat", action="store_true", default=True, help="Use UltraChat 200k dataset for high-quality conversations")
    parser.add_argument("--use_openorca", action="store_true", default=True, help="Use OpenOrca dataset for GPT-4 quality responses")
    parser.add_argument("--use_lmsys_chat", action="store_true", help="Use LMSYS Chat-1M dataset for real-world conversations")
    parser.add_argument("--use_chatbot_arena", action="store_true", help="Use Chatbot Arena dataset for human-preferred responses")
    parser.add_argument("--max_samples", type=int, default=25000, help="Maximum number of samples per individual dataset")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--ignore_checkpoint", action="store_true", help="Ignore existing checkpoints and start training from step 1")
    
    # Training hyperparameters  
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio of total steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    
    return parser.parse_args()

def validate_config(args):
    """Validate configuration and apply fixes."""
    # Handle mixed precision training flags
    if args.no_fp16:
        args.fp16 = False
    if args.bf16:
        args.fp16 = False  # BF16 takes precedence over FP16
    
    # Flash attention handling
    if args.no_flash_attention:
        args.use_flash_attention = False
    
    return args
