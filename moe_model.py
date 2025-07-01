import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from tutel import moe as tutel_moe

NUM_EXPERTS = 10


def create_moe_config(model_size: str = "500m", use_flash_attention: bool = True) -> GPT2Config:
    """
    Create GPT2Config for MoE model with flash attention support.
    
    Args:
        model_size: Model size configuration
        use_flash_attention: Whether to use flash attention for memory efficiency
        
    Returns:
        GPT2Config with appropriate parameters for MoE
    """
    
    # Base configuration parameters
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
    
    if model_size == "500m":
        # MoE 500M configuration (optimized for MoE)
        config = GPT2Config(
            n_positions=1024,     # Longer context for better conversations
            n_embd=1024,          # Hidden size
            n_layer=8,            # Number of layers (fewer than dense due to MoE efficiency)
            n_head=16,            # Attention heads
            n_inner=4096,         # FFN inner dimension (will be per-expert in MoE layers)
            **base_config
        )
        
    elif model_size == "1b":
        # MoE 1B configuration
        config = GPT2Config(
            n_positions=1024,     # Longer context
            n_embd=1280,          # Larger hidden size
            n_layer=12,           # More layers
            n_head=20,            # More attention heads
            n_inner=5120,         # Larger FFN dimension
            **base_config
        )
        
    elif model_size == "2b":
        # MoE 2B configuration
        config = GPT2Config(
            n_positions=2048,     # Even longer context
            n_embd=1536,          # Larger hidden size
            n_layer=16,           # More layers
            n_head=24,            # More attention heads
            n_inner=6144,         # Larger FFN dimension
            **base_config
        )
    
    else:
        raise ValueError(f"Unknown MoE model size: {model_size}. Supported: '500m', '1b', '2b'")
    
    return config


def get_moe_model_info(config: GPT2Config, num_experts: int = NUM_EXPERTS) -> dict:
    """
    Calculate MoE model parameter count and memory usage.
    
    Note: MoE models have more total parameters but fewer active parameters.
    """
    # Standard transformer parameters (same as dense model)
    embed_params = config.vocab_size * config.n_embd
    pos_embed_params = config.n_positions * config.n_embd
    
    # Attention parameters (per layer)
    attn_params_per_layer = 4 * config.n_embd * config.n_embd
    
    # MoE layer parameters calculation
    # Every other layer is MoE, others are standard FFN
    moe_layers = config.n_layer // 2
    dense_layers = config.n_layer - moe_layers
    
    # Dense FFN layers
    dense_ffn_params = dense_layers * (2 * config.n_embd * config.n_inner)
    
    # MoE FFN layers (num_experts * expert_size + gating)
    expert_params_per_layer = num_experts * (2 * config.n_embd * config.n_inner)
    gate_params_per_layer = config.n_embd * num_experts
    moe_ffn_params = moe_layers * (expert_params_per_layer + gate_params_per_layer)
    
    # Layer norms
    ln_params_per_layer = 2 * config.n_embd
    total_ln_params = ln_params_per_layer * config.n_layer
    
    # Final layer norm + LM head
    final_ln_params = config.n_embd
    lm_head_params = config.n_embd * config.vocab_size
    
    # Total parameters
    total_params = (embed_params + pos_embed_params + 
                   (attn_params_per_layer * config.n_layer) +
                   dense_ffn_params + moe_ffn_params + total_ln_params +
                   final_ln_params + lm_head_params)
    
    # Active parameters (assuming top-2 routing)
    k = 2  # top-k routing
    active_expert_ratio = k / num_experts
    active_moe_params = moe_layers * (
        (active_expert_ratio * expert_params_per_layer) + gate_params_per_layer
    )
    
    active_params = (embed_params + pos_embed_params +
                    (attn_params_per_layer * config.n_layer) +
                    dense_ffn_params + active_moe_params + total_ln_params +
                    final_ln_params + lm_head_params)
    
    # Memory usage (FP32)
    total_memory_mb = (total_params * 4) / (1024 * 1024)
    active_memory_mb = (active_params * 4) / (1024 * 1024)
    
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1_000_000,
        "active_params": active_params,
        "active_params_m": active_params / 1_000_000,
        "total_memory_fp32_mb": total_memory_mb,
        "total_memory_fp32_gb": total_memory_mb / 1024,
        "active_memory_fp32_mb": active_memory_mb,
        "active_memory_fp32_gb": active_memory_mb / 1024,
        "layers": config.n_layer,
        "moe_layers": moe_layers,
        "dense_layers": dense_layers,
        "hidden_size": config.n_embd,
        "attention_heads": config.n_head,
        "ffn_size": config.n_inner,
        "context_length": config.n_positions,
        "num_experts": num_experts,
        "experts_per_token": k,
        "expert_utilization": f"{active_expert_ratio*100:.1f}%"
    }

class MoEBlock(nn.Module):
    """Top‑2 MoE feed‑forward layer using Tutel"""
    def __init__(self, d_model, d_ff, num_experts, k=2):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.k = k
        
        # Define expert configuration according to Tutel's expected format
        expert_config = {
            'type': 'ffn',  # Specify the expert type as feed-forward network
            'count_per_node': self.num_experts,
            'hidden_size_per_expert': self.d_ff,
            'output_dim': d_model,  # Match output dim to model dim
            'activation_fn': lambda x: torch.nn.functional.gelu(x),  # Use callable activation
        }
        
        # Initialize MoE layer with load balancing configuration
        self.moe = tutel_moe.moe_layer(
            gate_type={
                'type': 'top', 
                'k': self.k,
                'fp32_gate': True,  # Use FP32 for gate computation for stability
                'gate_noise': 1e-2,  # Add noise to gate for better exploration
            },
            model_dim=self.d_model,
            experts=expert_config,
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds=(1, 1),
            batch_prioritized_routing=True,
            # Load balancing parameters that work with this Tutel version
            is_gshard_loss=True,  # Use GShard-style load balancing loss
        )
        
        # Store expert parameters for initialization
        self.experts = self.moe.experts

    def forward(self, x):
        # Store input shape for diagnostics
        batch_size, seq_len, d_model = x.shape
        
        # Forward through MoE
        output = self.moe(x)
        
        # Periodically log routing info (every 1000 forward passes)
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        
        if self._forward_count % 1000 == 0:
            try:
                # Try to extract routing info from Tutel's gate
                if hasattr(self.moe, 'gate') and hasattr(self.moe.gate, 'wg'):
                    gate_logits = self.moe.gate.wg.weight.data
                    # Simple check: are gate weights diverse?
                    gate_std = gate_logits.std().item()
                    gate_mean = gate_logits.mean().item()
                    print(f"MoE Gate Stats: mean={gate_mean:.4f}, std={gate_std:.4f}, tokens={batch_size*seq_len}")
            except:
                pass  # Ignore errors in diagnostics
        
        return output
        
    def init_from_mlp(self, mlp):
        """Initialize MoE experts from a standard MLP layer"""
        with torch.no_grad():
            if hasattr(self.experts, 'experts'):
                # Handle case where experts is a wrapper object
                experts = self.experts.experts
            else:
                # Handle case where experts is directly accessible
                experts = [self.experts]
                
            for expert in experts:
                if hasattr(expert, 'fc1'):
                    expert.fc1.weight.copy_(mlp.c_fc.weight.data)
                    if hasattr(mlp.c_fc, 'bias') and expert.fc1.bias is not None:
                        expert.fc1.bias.copy_(mlp.c_fc.bias.data)
                    expert.fc2.weight.copy_(mlp.c_proj.weight.data)
                    if hasattr(mlp.c_proj, 'bias') and expert.fc2.bias is not None:
                        expert.fc2.bias.copy_(mlp.c_proj.bias.data)

class GPT2WithMoE(GPT2PreTrainedModel, GenerationMixin):
    """GPT‑2 backbone with MoE layers and flash attention support"""
    def __init__(self, config: GPT2Config, num_experts: int = NUM_EXPERTS):
        config_class = type(config)
        # Initialize with the parent class
        super().__init__(config)
        
        # Store configuration
        self.num_experts = num_experts
        self.config_class = config_class
        
        # Check if flash attention is enabled
        self.use_flash_attention = getattr(config, 'attn_implementation', None) == 'sdpa'
        
        # Initialize base model with flash attention support
        if self.use_flash_attention:
            print(f"✓ Flash attention (SDPA) enabled for MoE model")
        
        self.transformer = AutoModelForCausalLM.from_config(config).transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

        # Initialize MoE layers (replace every other FFN layer with MoE)
        moe_layer_count = 0
        for idx, block in enumerate(self.transformer.h):
            if idx % 2 == 0:  # Replace even-indexed layers with MoE
                # Store the original MLP weights for initialization
                original_mlp = block.mlp
                moe_block = MoEBlock(
                    d_model=config.n_embd,
                    d_ff=config.n_inner,
                    num_experts=self.num_experts,
                    k=2
                )
                # Initialize MoE experts with the original MLP weights
                moe_block.init_from_mlp(original_mlp)
                block.mlp = moe_block
                moe_layer_count += 1
        
        print(f"✓ Initialized {moe_layer_count} MoE layers with {self.num_experts} experts each")
        
        # Apply final processing again after MoE initialization
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get logits from the language model head
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # Don't compute loss here - let CustomTrainer.compute_loss handle it
        # This ensures MoE auxiliary loss is properly included
        loss = None
        
        # Return a CausalLMOutputWithCrossAttentions object
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Only keep the last token for generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
        
    def generate(self, input_ids=None, attention_mask=None, **generation_kwargs):
        # Set default generation parameters if not provided
        default_kwargs = {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.config.eos_token_id,
            "eos_token_id": self.config.eos_token_id,
        }
        
        # Update with any user-provided kwargs
        generation_kwargs = {**default_kwargs, **generation_kwargs}
        
        # Call the parent class's generate method
        return GenerationMixin.generate(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
