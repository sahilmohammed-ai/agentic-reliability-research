"""
the verifier model: a value head on top of a qwen backbone.

architecture:
  - base: qwen2.5-3b-instruct, loaded via AutoModel (no lm head, we don't generate text).
  - head: a single linear layer mapping the last token's hidden state to 2 outputs,
    q_value and advantage. this mirrors agentprm's design (value estimator, not a
    classifier), just with plain q-value/advantage naming instead of promise/progress.

the backbone can be frozen or fully fine-tuned, controlled by freeze_backbone:
  - frozen: only the head trains. cheap enough to run locally on this machine's mps
    memory (~20gb unified), useful for a quick correctness check of the training loop.
  - unfrozen: the whole 3b model trains alongside the head, same as agentprm's own
    setup. this needs real gpu memory (a100/h100-class), so it runs on lightning ai,
    not locally. this is the mode used for the actual verifier checkpoint.
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

        hidden_size = self.backbone.config.hidden_size
        # two outputs: q_value and advantage, both plain scalars (no activation, these
        # are regression targets that can be negative for advantage). kept in float32
        # since this is the part of the model most sensitive to precision during the
        # loss computation, even when the backbone itself trains in bfloat16.
        self.value_head = nn.Linear(hidden_size, 2, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """returns a (batch, 2) tensor: column 0 = q_value, column 1 = advantage."""
        # no_grad only when frozen, otherwise we need the autograd graph through the backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # pull the hidden state at each sequence's last real (non-padded) token, since
        # that's the point where the model has seen the full input and should output
        # its estimate for the state described by that turn
        seq_lengths = attention_mask.sum(dim=1) - 1  # index of last real token per row
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        return self.value_head(last_token_hidden.float())
