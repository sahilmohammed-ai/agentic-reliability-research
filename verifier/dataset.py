"""
turns labeled trajectory jsons into a flat (text, q_value, advantage) dataset for the verifier.

each worker turn becomes one training example. the input text mirrors what the worker agent
itself sees when choosing an action (task, plan, observation, action taken), so the verifier
learns to score the same state/action context the agents act on.
"""

import glob
import json
import os

import torch
from torch.utils.data import Dataset


def build_input_text(task_goal: str, plan: str, obs_before: str, action: str) -> str:
    """same shape as the worker's prompt, so the verifier scores what the worker actually saw."""
    return (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Observation:\n{obs_before}\n\n"
        f"Action taken: {action}"
    )


def load_examples(labeled_dir: str) -> list[dict]:
    """flatten every worker turn across all trajectory files into one list of examples."""
    examples = []
    for path in sorted(glob.glob(os.path.join(labeled_dir, "*.json"))):
        with open(path) as f:
            traj = json.load(f)

        for turn in traj["turns"]:
            if turn["role"] != "worker":
                continue
            # label.py only adds q_value/advantage to worker turns, so this should always be present
            if "q_value" not in turn:
                continue

            examples.append({
                "text": build_input_text(traj["task_goal"], traj["plan"], turn["obs_before"], turn["action"]),
                "q_value": turn["q_value"],
                "advantage": turn["advantage"],
            })

    return examples


class VerifierDataset(Dataset):
    """tokenizes examples lazily so the whole dataset doesn't need to fit in memory at once."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            # float32 targets: these only ever meet the value head, which trains in float32
            "q_value": torch.tensor(ex["q_value"], dtype=torch.float32),
            "advantage": torch.tensor(ex["advantage"], dtype=torch.float32),
        }

# copy tokens into padded tensors
def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """right-pad input_ids/attention_mask to the longest sequence in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    q_values = torch.stack([item["q_value"] for item in batch])
    advantages = torch.stack([item["advantage"] for item in batch])

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "q_value": q_values,
        "advantage": advantages,
    }
