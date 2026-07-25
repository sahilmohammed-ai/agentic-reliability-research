"""
coordinator policy: qwen backbone + 3-way categorical head over {continue, retry, replan}, trained
with PPO against verifier_v5's ADVANTAGE (q(t) - q(t-1) on the resulting turn, not raw q_value --
see coordinator/train_ppo.py's collect_batch) as the reward. same backbone-plus-small-head pattern
as verifier/model.py, but a policy head (softmax over discrete actions) instead of a 2-scalar
regression head -- the critic is NOT trained here, PPO reuses the already-validated verifier_v5 as
a frozen critic instead (see train_ppo.py's docstring for why).

3-way, not 4 or 5: an independent code review (Opus advisor pass) found "backtrack" was
implemented identically to "retry" (both just masked the last action) -- a real bug that would have
made those two actions behaviorally indistinguishable, wasting a class in the policy head and
making any later "the coordinator learned to use backtrack" claim uninterpretable. rather than
invent new backtrack semantics not validated anywhere in this project (the original deleted
coordinator.py's "clear history + fresh plan" design was never tested against the alternatives),
backtrack was dropped. continue/retry/replan are the three genuinely distinct tools: no-op, small
local correction (mask+nudge), and large correction (revise the plan).

honest framing, not glossed over: this reward is a PROXY, not exact credit assignment for the
coordinator's decision. the verifier's advantage was trained/validated to score WORKER-turn
quality (build 10-12), not coordinator-decision quality directly. the coordinator's action does
causally shape the resulting turn (replan changes the plan the worker acts on, backtrack changes
which commands are available), so crediting it with that turn's verifier score is a reasonable
1-step lookahead -- but it means the coordinator is optimizing "make the verifier score the next
turn well," which could in principle diverge from "coordinate well" if the verifier has any
exploitable blind spot. worth stating plainly in any writeup, not something this design eliminates.

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
ACTIONS = ("continue", "retry", "replan")


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
