"""
verifier: qwen backbone + value head for q-value and advantage prediction.
frozen backbone trains only the head (local testing). unfrozen trains full model (needs gpu).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

# switched from Qwen2.5-3B-Instruct to 1.5B for verifier v2 (2026-07-17): the verifier only
# outputs 2 scalars per turn, and inference cost scales with backbone size regardless of
# frozen/unfrozen (freeze_backbone only cuts TRAINING cost, not inference FLOPs -- a distinction
# that had this deferred until now). hidden_size is read dynamically from the backbone's own
# config below, so the value head resizes automatically (2048 -> 1536), no other code changes
# needed. not yet validated that percentile separation (won vs lost turns) holds at this smaller
# size -- that's the required check before trusting this checkpoint, see .info/CLAUDE.md.
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
        """return (batch, 2) tensor: [q_value, advantage].

        q_value is passed through sigmoid, bounding it to (0, 1) to match the training target's
        actual range (a discounted return, 1.0=certain win, 0.0=certain loss). added in verifier
        v2 after finding the previous unbounded linear output had no architectural pressure to
        stay in range: live inference on real data had already produced q_values outside [0,1]
        (e.g. 1.795, 1.072, -0.032), which are meaningless under a "probability of eventual
        success" interpretation. advantage is left unbounded (no sigmoid): it's a signed delta
        (can be negative, a turn that hurt progress), not a probability, so a (0,1) squashing
        function is not appropriate for it."""
        # only compute gradients when unfrozen and training (avoid val OOM)
        needs_grad = (not self.freeze_backbone) and self.training
        with torch.set_grad_enabled(needs_grad):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # extract last real token's hidden state (model has seen full context)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        raw = self.value_head(last_token_hidden.float())
        q_value = torch.sigmoid(raw[:, 0:1])
        advantage = raw[:, 1:2]
        return torch.cat([q_value, advantage], dim=1)
