import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from tutel import moe as tutel_moe

NUM_EXPERTS = 16
class MoEBlock(nn.Module):
    """Top‑2 MoE feed‑forward layer using Tutel"""
    def __init__(self, d_model, d_ff, num_experts=NUM_EXPERTS, k=2):
        super().__init__()
        self.moe = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': k},
            model_dim=d_model,
            experts={
                'num_experts_per_device': num_experts,
                'type': 'ffn',
                'hidden_size_per_expert': d_ff,
                'activation_fn': lambda x: torch.nn.functional.gelu(x)
            }
        )

    def forward(self, x):
        return self.moe(x)

class GPT2WithMoE(GPT2PreTrainedModel, GenerationMixin):
    """GPT‑2 backbone where every second MLP is replaced by MoEBlock"""
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        base = AutoModelForCausalLM.from_config(config)
        self.transformer = base.transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        for idx, block in enumerate(self.transformer.h):
            if idx % 2 == 0:
                block.mlp = MoEBlock(
                    d_model=config.n_embd,
                    d_ff=config.n_inner,
                    num_experts=NUM_EXPERTS,
                    k=2
                )
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
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
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
