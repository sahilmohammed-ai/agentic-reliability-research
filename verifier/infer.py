"""
live verifier inference: load a trained checkpoint and score a single turn's q_value/advantage
in real time. no training loop, no gradients, no dataloader, just one forward pass per call.

score() and score_batch() give a live per-turn signal for anything that needs to react to a
rollout in progress (e.g. best-of-n / verifier-guided search), independent of any particular
coordination mechanism consuming it.
"""

import os

import torch
from transformers import AutoTokenizer

from verifier.dataset import build_input_text
from verifier.model import VerifierModel, BASE_MODEL


def get_device() -> torch.device:
    """prefer cuda, then apple silicon mps, then cpu. same preference order as verifier/train.py."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Verifier:
    """loads a trained verifier checkpoint once, then scores turns cheaply.

    usage:
        verifier = Verifier("checkpoints/verifier_v2")
        q_value, advantage = verifier.score(task_goal, plan, obs_before, action)
    """

    def __init__(self, checkpoint_dir: str, max_length: int = 256, device: torch.device | None = None):
        self.device = device or get_device()
        self.max_length = max_length

        checkpoint_path = os.path.join(checkpoint_dir, "verifier.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"no verifier.pt found in {checkpoint_dir}. train the verifier first "
                f"(verifier/train.py) and place the checkpoint here."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # the checkpoint was trained with freeze_backbone=False (full fine-tune), so the saved
        # state_dict includes the fine-tuned backbone weights, not just the head. freeze_backbone
        # here only controls the forward pass's grad-tracking branch, irrelevant at inference
        # since everything runs under no_grad regardless.
        self.model = VerifierModel(BASE_MODEL, freeze_backbone=True)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, task_goal: str, plan: str, obs_before: str, action: str) -> tuple[float, float]:
        """returns (q_value, advantage) for one turn, same input shape used in training."""
        text = build_input_text(task_goal, plan, obs_before, action)
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        prediction = self.model(encoded["input_ids"], encoded["attention_mask"])
        q_value, advantage = prediction[0, 0].item(), prediction[0, 1].item()
        return q_value, advantage

    @torch.no_grad()
    def score_batch(
        self, items: list[tuple[str, str, str, str]]
    ) -> list[tuple[float, float]]:
        """batched version of score(), for scoring several candidate turns at once
        (e.g. best-of-n / verifier-guided search). each item is (task_goal, plan, obs_before, action)."""
        texts = [build_input_text(*item) for item in items]
        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(self.device)

        predictions = self.model(encoded["input_ids"], encoded["attention_mask"])
        return [(row[0].item(), row[1].item()) for row in predictions]
