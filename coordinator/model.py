"""
coordinator policy: qwen backbone + 4-way categorical head over
{continue, retry, replan, backtrack}, trained with PPO against verifier_v5's q_value
as the reward (see coordinator/train_ppo.py). same backbone-plus-small-head pattern as
verifier/model.py, but a policy head (softmax over discrete actions) instead of a 2-scalar
regression head -- the critic is NOT trained here, PPO reuses the already-validated verifier_v5
as a frozen critic instead (see train_ppo.py's docstring for why).

no "escalate" action: the original (ALFWorld-era, deleted) coordinator.py had a 5th action that
swapped in a stronger model for one turn. build 11's diagnosis (.info/BUILD_11_12_DIAGNOSIS.md)
found this confounds the win-rate signal (68.2% of all turns escalated, 71% escalate rate in LOST
episodes vs 30% in won -- heavy escalation mostly did not rescue a failing trajectory) and breaks
the project's "frozen role agents, only the coordinator learns" framing (.info/CLAUDE.md) by
letting the worker model itself change mid-episode. every turn's verifier reward here reflects the
SAME frozen Qwen2.5-3B worker throughout, so PPO learns pure coordination skill, not partly "is a
stronger model available this turn."
"""

import torch
import torch.nn as nn
from transformers import AutoModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTIONS = ("continue", "retry", "replan", "backtrack")


class CoordinatorModel(nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, freeze_backbone: bool = False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = AutoModel.from_pretrained(base_model, dtype=torch.bfloat16)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False

        hidden_size = self.backbone.config.hidden_size
        # policy head outputs one logit per discrete action (softmax applied by the caller,
        # kept as raw logits here so both sampling and cross-entropy-style PPO losses can use them)
        self.policy_head = nn.Linear(hidden_size, len(ACTIONS), dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """return (batch, len(ACTIONS)) raw logits over the discrete action set."""
        needs_grad = (not self.freeze_backbone) and self.training
        with torch.set_grad_enabled(needs_grad):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        return self.policy_head(last_token_hidden.float())
