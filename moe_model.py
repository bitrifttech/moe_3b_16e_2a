import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel, AutoModelForCausalLM
import tutel

class MoEBlock(nn.Module):
    """Top‑2 MoE feed‑forward layer using Tutel"""
    def __init__(self, d_model, d_ff, num_experts=16, k=2):
        super().__init__()
        self.gate = tutel.moe.TopKGate(d_model, num_experts, k=k)
        self.experts = tutel.moe.MExperts([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        y, _ = tutel.moe.moe_layer.apply(self.gate, self.experts, x)
        return y

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
                    num_experts=16,
                    k=2
                )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}
