"""
Model initialization and setup for training.
"""
import os
import json
import torch
import logging
from transformers import GPT2Config, AutoTokenizer
from datasets import Dataset

# Import model classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from moe_model import GPT2WithMoE
from dense_model import GPT2Dense, create_dense_config

# Get logger
logger = logging.getLogger(__name__)

def setup_tokenizer():
    """Setup and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_model(args, tokenizer):
    """Create model based on architecture choice."""
    if args.architecture == "moe":
        return _create_moe_model(args, tokenizer)
    elif args.architecture == "dense":
        return _create_dense_model(args, tokenizer)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

def _create_moe_model(args, tokenizer):
    """Create MoE model configuration."""
    print(f"Creating MoE model architecture...")
    
    # MoE model configuration - existing configuration
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=max(1024, args.max_length),  # Use longer context length
        n_embd=1024,  # Increased model capacity
        n_layer=8,  # Increased from 6
        n_head=16,  # Increased from 12
        n_inner=4096,  # Increased feed-forward dimension
        n_experts=16,  # Number of experts
        expert_top_k=2,  # Top-k routing
        moe_layers=[2, 4, 6],  # Which layers to use MoE
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"MoE Config: {cfg.n_layer} layers, {cfg.n_embd} embed dim, {cfg.n_experts} experts")
    
    # Create MoE model
    model = GPT2WithMoE(cfg)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MoE Model - Total parameters: {total_params:,}")
    print(f"MoE Model - Trainable parameters: {trainable_params:,}")
    
    return model, cfg

def _create_dense_model(args, tokenizer):
    """Create dense model configuration."""
    print(f"Creating {args.model_size} dense model architecture...")
    
    # Create dense model configuration
    cfg = create_dense_config(
        model_size=args.model_size,
        vocab_size=len(tokenizer),
        max_length=args.max_length,
        use_flash_attention=args.use_flash_attention
    )
    
    print(f"Dense Config: {cfg.n_layer} layers, {cfg.n_embd} embed dim, {cfg.n_head} heads")
    
    # Create dense model
    model = GPT2Dense(cfg)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Dense Model ({args.model_size}) - Total parameters: {total_params:,}")
    print(f"Dense Model ({args.model_size}) - Trainable parameters: {trainable_params:,}")
    
    return model, cfg

def setup_output_directory(args):
    """Setup output directory based on architecture."""
    if args.architecture == "moe":
        output_dir = f"{args.output_dir}_moe"
    else:  # dense
        output_dir = f"{args.output_dir}_dense_{args.model_size}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_model_config(output_dir, cfg, args):
    """Save model configuration to file."""
    model_info = {
        "architecture": args.architecture,
        "model_size": getattr(args, 'model_size', 'N/A'),
        "max_length": args.max_length,
        "use_flash_attention": getattr(args, 'use_flash_attention', False),
        "gradient_checkpointing": args.gradient_checkpointing,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "config": cfg.to_dict() if hasattr(cfg, 'to_dict') else str(cfg)
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model configuration saved to: {config_path}")
    return config_path
