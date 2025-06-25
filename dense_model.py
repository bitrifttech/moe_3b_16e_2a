import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM


class GPT2Dense(GPT2LMHeadModel):
    """
    Standard Dense GPT-2 model with configurable size.
    Simple, reliable transformer architecture without MoE complexity.
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        
        # Store config for proper serialization
        self.config_class = type(config)
        
        # The model is already initialized by the parent class
        # No additional modifications needed for dense architecture
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Standard forward pass for causal language modeling.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs


def create_dense_config(model_size: str = "800m", use_flash_attention: bool = True) -> GPT2Config:
    """
    Create GPT2Config for different dense model sizes.
    
    Args:
        model_size: One of "200m", "400m", "600m", "800m", "1.5b", "medium", "large"
        use_flash_attention: Whether to use flash attention for memory efficiency
        
    Returns:
        GPT2Config with appropriate parameters
    """
    
    # Base configuration parameters that are common
    base_config = {
        "vocab_size": 50257,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "use_cache": True,
    }
    
    # Add flash attention if supported and requested
    if use_flash_attention:
        # Use PyTorch's built-in scaled dot product attention (available in PyTorch 2.0+)
        # This provides memory efficiency without requiring the flash-attn package
        base_config.update({
            "attn_implementation": "sdpa",  # Use PyTorch's scaled_dot_product_attention
        })
    
    if model_size.lower() == "200m":
        # 200M parameter model (smaller than GPT2-Medium)
        config = GPT2Config(
            n_positions=1024,     # Shorter context for memory efficiency
            n_embd=768,           # Hidden size
            n_layer=16,           # Number of layers
            n_head=12,            # Attention heads
            n_inner=3072,         # FFN inner dimension (4 * n_embd)
            **base_config
        )
        
    elif model_size.lower() == "400m":
        # 400M parameter model 
        config = GPT2Config(
            n_positions=1536,     # Medium context length
            n_embd=960,           # Hidden size
            n_layer=24,           # Number of layers
            n_head=15,            # Attention heads (960 / 64 = 15)
            n_inner=3840,         # FFN inner dimension (4 * n_embd)
            **base_config
        )
        
    elif model_size.lower() == "600m":
        # 600M parameter model
        config = GPT2Config(
            n_positions=1536,     # Medium context length
            n_embd=1152,          # Hidden size
            n_layer=24,           # Number of layers
            n_head=18,            # Attention heads (1152 / 64 = 18)
            n_inner=4608,         # FFN inner dimension (4 * n_embd)
            **base_config
        )
        
    elif model_size.lower() in ["800m", "medium+"]:
        # 800M parameter model (GPT2-Medium+)
        config = GPT2Config(
            n_positions=2048,     # Context length
            n_embd=1280,          # Hidden size
            n_layer=36,           # Number of layers
            n_head=20,            # Attention heads
            n_inner=5120,         # FFN inner dimension (4 * n_embd)
            **base_config
        )
        
    elif model_size.lower() in ["1.5b", "large+"]:
        # 1.5B parameter model (GPT2-Large+)
        config = GPT2Config(
            n_positions=2048,     # Context length
            n_embd=1600,          # Hidden size
            n_layer=48,           # Number of layers
            n_head=25,            # Attention heads (1600 / 64 = 25)
            n_inner=6400,         # FFN inner dimension (4 * n_embd)
            **base_config
        )
        
    elif model_size.lower() == "medium":
        # Standard GPT2-Medium (355M)
        config = GPT2Config(
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            **base_config
        )
        
    elif model_size.lower() == "large":
        # Standard GPT2-Large (774M)
        config = GPT2Config(
            n_positions=1024,
            n_embd=1280,
            n_layer=36,
            n_head=20,
            n_inner=5120,
            **base_config
        )
    
    else:
        raise ValueError(f"Unknown model size: {model_size}. Supported: '200m', '400m', '600m', '800m', '1.5b', 'medium', 'large'")
    
    return config


def get_model_info(config: GPT2Config) -> dict:
    """
    Calculate model parameter count and memory usage.
    """
    # Embedding parameters
    embed_params = config.vocab_size * config.n_embd  # Token embeddings
    pos_embed_params = config.n_positions * config.n_embd  # Position embeddings
    
    # Transformer layer parameters (per layer)
    # Attention: 4 * n_embd^2 (Q, K, V, O projections)
    attn_params_per_layer = 4 * config.n_embd * config.n_embd
    # Feed-forward: 2 * n_embd * n_inner (up + down projections)
    ffn_params_per_layer = 2 * config.n_embd * config.n_inner
    # Layer norms: 2 * n_embd (pre-attn + pre-ffn)
    ln_params_per_layer = 2 * config.n_embd
    
    layer_params = (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer)
    total_layer_params = layer_params * config.n_layer
    
    # Final layer norm + LM head
    final_ln_params = config.n_embd
    lm_head_params = config.n_embd * config.vocab_size
    
    # Total parameters
    total_params = (embed_params + pos_embed_params + total_layer_params + 
                   final_ln_params + lm_head_params)
    
    # Memory usage (FP32)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per FP32 parameter
    memory_gb = memory_mb / 1024
    
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1_000_000,
        "memory_fp32_mb": memory_mb,
        "memory_fp32_gb": memory_gb,
        "layers": config.n_layer,
        "hidden_size": config.n_embd,
        "attention_heads": config.n_head,
        "ffn_size": config.n_inner,
        "context_length": config.n_positions,
    }


if __name__ == "__main__":
    # Test different configurations
    for size in ["medium", "large", "200m", "400m", "600m", "800m", "1.5b"]:
        print(f"\n=== {size.upper()} Configuration ===")
        config = create_dense_config(size)
        info = get_model_info(config)
        
        print(f"Parameters: {info['total_params_m']:.1f}M")
        print(f"Layers: {info['layers']}")
        print(f"Hidden size: {info['hidden_size']}")
        print(f"Attention heads: {info['attention_heads']}")
        print(f"FFN size: {info['ffn_size']}")
        print(f"Context: {info['context_length']}")
        print(f"Memory (FP32): {info['memory_fp32_gb']:.2f} GB")
