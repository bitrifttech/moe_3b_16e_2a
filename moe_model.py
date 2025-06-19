import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from tutel import moe as tutel_moe

NUM_EXPERTS = 16

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
    """GPT‑2 backbone where every second MLP is replaced by MoEBlock"""
    def __init__(self, config: GPT2Config):
        config_class = type(config)
        # Initialize with the parent class
        super().__init__(config)
        
        # Initialize base model
        self.transformer = AutoModelForCausalLM.from_config(config).transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
        
        # Store the config class for proper serialization
        self.config_class = config_class

        # Initialize MoE layers
        for idx, block in enumerate(self.transformer.h):
            if idx % 2 == 0:
                # Store the original MLP weights for initialization
                original_mlp = block.mlp
                moe_block = MoEBlock(
                    d_model=config.n_embd,
                    d_ff=config.n_inner,
                    num_experts=NUM_EXPERTS,
                    k=2
                )
                # Initialize MoE experts with the original MLP weights
                moe_block.init_from_mlp(original_mlp)
                block.mlp = moe_block
        
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
