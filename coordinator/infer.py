"""
live coordinator inference: load a policy checkpoint (trained or freshly initialized) and choose
an intervention action for one turn, either by sampling (needed during PPO rollout collection, so
log_prob is available for the policy-gradient update) or greedily (needed for eval/deployment).

input text mirrors the verifier's build_input_text so the coordinator sees the same task/plan/obs
context the verifier scores, plus the verifier's own q_value/advantage appended -- the coordinator
needs to see the reward signal it's being trained against, not just raw environment text.
"""

import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from coordinator.model import ACTIONS, BASE_MODEL, CoordinatorModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_coordinator_input_text(
    task_goal: str, plan: str, obs_before: str, action_history: list[str],
    q_value: float, advantage: float,
) -> str:
    """state representation the coordinator policy conditions on: task/plan/current-obs (same
    shape the worker/verifier see) plus recent action history (for repetition/stall context) and
    the verifier's live q_value/advantage for the turn just taken (the actual reward signal the
    coordinator is being trained to respond to)."""
    recent = " -> ".join(action_history[-5:]) if action_history else "(none yet)"
    return (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Observation:\n{obs_before}\n\n"
        f"Recent actions: {recent}\n"
        f"Verifier q_value: {q_value:.3f}, advantage: {advantage:.3f}"
    )


class CoordinatorPolicy:
    """loads a coordinator checkpoint (or a fresh untrained backbone for iteration 0 of PPO) and
    chooses actions.

    usage:
        policy = CoordinatorPolicy("checkpoints/coordinator_v0")   # or None for a fresh init
        action, log_prob = policy.sample(task_goal, plan, obs_before, action_history, q_value, advantage)
        action = policy.act_greedy(...)   # deterministic, for eval
    """

    def __init__(
        self,
        checkpoint_dir: str | None,
        max_length: int = 384,
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length

        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "coordinator.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"no coordinator.pt found in {checkpoint_dir}. train first "
                    f"(coordinator/train_ppo.py) or pass checkpoint_dir=None for a fresh policy."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.model = CoordinatorModel(BASE_MODEL, freeze_backbone=True)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.model = CoordinatorModel(BASE_MODEL, freeze_backbone=True)
        # forced explicitly -- see coordinator/train_ppo.py's identical comment. the model reads
        # the LAST non-pad token, which is only correct under right-padding.
        self.tokenizer.padding_side = "right"

        self.model.to(self.device)
        self.model.eval()

    def _logits(
        self, task_goal: str, plan: str, obs_before: str, action_history: list[str],
        q_value: float, advantage: float,
    ) -> torch.Tensor:
        text = build_coordinator_input_text(task_goal, plan, obs_before, action_history, q_value, advantage)
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        return self.model(encoded["input_ids"], encoded["attention_mask"])[0]

    @torch.no_grad()
    def sample(
        self, task_goal: str, plan: str, obs_before: str, action_history: list[str],
        q_value: float, advantage: float,
    ) -> tuple[str, float]:
        """stochastic action choice, needed during PPO rollout collection so the sampled action's
        log_prob under the CURRENT policy can be stored for the clipped-ratio update later.
        returns (action_name, log_prob)."""
        logits = self._logits(task_goal, plan, obs_before, action_history, q_value, advantage)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return ACTIONS[idx.item()], dist.log_prob(idx).item()

    @torch.no_grad()
    def act_greedy(
        self, task_goal: str, plan: str, obs_before: str, action_history: list[str],
        q_value: float, advantage: float,
    ) -> str:
        """deterministic argmax choice, for evaluation/deployment (not training rollouts)."""
        logits = self._logits(task_goal, plan, obs_before, action_history, q_value, advantage)
        return ACTIONS[torch.argmax(logits).item()]
