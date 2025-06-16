import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel, AutoModelForCausalLM
from tutel import moe as tutel_moe

NUM_EXPERTS = 16  # Match build plan: 16 experts per device

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

class GPT2WithMoE(GPT2PreTrainedModel):
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
        out = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}
