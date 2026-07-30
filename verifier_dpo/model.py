"""
verifier_dpo: episode-pair preference model, kept fully separate from verifier/ (v1-v5).

standing problem this attempts (see .info/CLAUDE.md): every relabeling scheme tried so far
(MC return-to-go, TD/GAE, agent_prm's shaped target, branching-rollout MC returns) inherits the
same ceiling -- TextWorldExpress gives ~0 reward on almost every non-terminal turn in these games,
so no per-step regression target can carry real same-state action-discrimination signal. this
model does NOT attempt to fix that (confirmed not fixable by relabeling alone). instead it takes
a structurally different form of supervision: real per-EPISODE outcomes (won/lost), which are
NOT sparse -- every collected episode has exactly one, always known.

architecture: same qwen backbone as verifier/model.py, but the head outputs ONE scalar per TOKEN
POSITION (not one pooled scalar for the whole sequence). training uses a trajectory-level
preference loss (bradley-terry / DPO-style, see train.py): push the model to prefer a real winning
episode's full text over a real losing episode's full text, for the same game. because
log P(episode) = sum over tokens of log P(token | prefix) under a causal LM view of the score head
as an unnormalized log-density, the SAME per-token scores that produce the trajectory-level
preference can be read off individually afterward and pooled per-turn (mean over each turn's action
tokens) -- a real per-turn score, but one that reflects "was this turn part of a winning-flavored
trajectory," not "was this specific action better than the untaken alternative at this state." see
this package's train.py module docstring for what this can and cannot be expected to show.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class PreferenceScorer(nn.Module):
    """backbone + per-token scalar head. forward() returns one score per input token position,
    so callers can both (a) sum/mean them for a whole-episode preference loss and (b) slice out
    the token span belonging to one turn's action for a per-turn read-out."""

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
        self.score_head = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """return (batch, seq_len) per-token scores. padded positions are meaningless (caller
        must mask with attention_mask before pooling)."""
        needs_grad = (not self.freeze_backbone) and self.training
        with torch.set_grad_enabled(needs_grad):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        scores = self.score_head(hidden_states.float()).squeeze(-1)  # (batch, seq_len)
        return scores

    def episode_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """sum of per-token scores over real (non-pad) tokens -- the trajectory-level score used
        directly in the preference loss. sum (not mean) matches the log P(episode) = sum_t
        log P(token_t) identity the per-turn read-out in infer.py relies on."""
        scores = self.forward(input_ids, attention_mask)
        return (scores * attention_mask.float()).sum(dim=1)
