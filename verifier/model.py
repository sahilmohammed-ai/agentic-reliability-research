"""
verifier: qwen backbone + value head for q-value and advantage prediction.
frozen backbone trains only the head (local testing). unfrozen trains full model (needs gpu).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class VerifierModel(nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, freeze_backbone: bool = False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = AutoModel.from_pretrained(base_model, dtype=torch.bfloat16)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            # gradient checkpointing trades compute for memory during backprop
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False

        hidden_size = self.backbone.config.hidden_size
        # value head outputs q_value and advantage (regression, kept in float32)
        self.value_head = nn.Linear(hidden_size, 2, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """return (batch, 2) tensor: [q_value, advantage]."""
        # only compute gradients when unfrozen and training (avoid val OOM)
        needs_grad = (not self.freeze_backbone) and self.training
        with torch.set_grad_enabled(needs_grad):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # extract last real token's hidden state (model has seen full context)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        return self.value_head(last_token_hidden.float())
