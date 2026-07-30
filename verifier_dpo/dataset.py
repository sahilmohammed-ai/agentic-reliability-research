"""
loads labeled trajectory jsons (same files verifier/dataset.py reads) but builds WHOLE-EPISODE
text blocks and pairs a won episode against a lost episode of the SAME game, instead of flattening
to per-turn (text, q_value, advantage) examples. no q_value/advantage labels are used at all --
the only supervision is each episode's real won/lost outcome.
"""

import glob
import json
import os
import random

import torch
from torch.utils.data import Dataset


def build_episode_text(traj: dict) -> tuple[str, list[tuple[int, int]]]:
    """concatenate every worker turn into one block, as ONE continuous sequence for the whole
    episode (not separate per-turn examples). returns (text, turn_char_spans), where
    turn_char_spans[i] = (start_char, end_char) of turn i's OWN block (observation + action) within
    the returned text -- used by infer.py to slice per-turn scores out of a single whole-episode
    forward pass, rather than re-encoding each turn in isolation.

    fixes a real train/inference mismatch caught in review (2026-07-29): infer.py previously
    re-encoded each turn as an ISOLATED (task, plan, obs, action) string via a separate
    build_prefix_text() function with different separators and the plan re-attached to every
    turn -- text the model never saw during training, since training only ever sees the whole
    episode in this function's format. slicing spans from one real encoding removes the
    mismatch entirely instead of trying to hand-match two separately-tokenized text formats."""
    header = f"Task: {traj['task_goal']}\n\nPlan:\n{traj['plan']}\n"
    text = header
    spans: list[tuple[int, int]] = []
    for turn in traj["turns"]:
        if turn["role"] != "worker":
            continue
        block = (
            f"\n[Turn {turn['step']}]\n"
            f"Observation:\n{turn['obs_before']}\n"
            f"Action taken: {turn['action']}"
        )
        start = len(text)
        text += block
        spans.append((start, len(text)))
    return text, spans


def load_episodes_by_game(labeled_dir: str) -> dict[str, dict[str, list[dict]]]:
    """returns {game: {"won": [episode_dict, ...], "lost": [episode_dict, ...]}}.

    game is inferred from the filename prefix (e.g. "coin_train_0000.json" -> "coin"), matching
    the naming convention already used across data/labeled/*."""
    by_game: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(glob.glob(os.path.join(labeled_dir, "*.json"))):
        fname = os.path.basename(path)
        game = fname.split("_")[0]
        with open(path) as f:
            traj = json.load(f)

        worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
        if not worker_turns:
            continue

        text, turn_spans = build_episode_text(traj)
        entry = {
            "episode_id": fname,
            "text": text,
            "turn_spans": turn_spans,
            "turns": worker_turns,
            "won": bool(traj["won"]),
            "n_turns": len(worker_turns),
        }
        bucket = by_game.setdefault(game, {"won": [], "lost": []})
        bucket["won" if entry["won"] else "lost"].append(entry)

    return by_game


def split_by_game(
    by_game: dict[str, dict[str, list[dict]]], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[dict[str, dict[str, list[dict]]], dict[str, dict[str, list[dict]]]]:
    """episode-grouped train/val split, done PER GAME and PER OUTCOME so both splits keep a
    representative won/lost mix of every game, then pairs are built separately within each split
    (see make_pairs()) so no episode's text appears in both a training pair and a validation pair.

    added after review (2026-07-29): the first real run had no held-out split at all --
    pairwise_acc was computed on the exact same pairs used for training, which is pure in-sample
    fit and cannot distinguish real generalization from memorizing a handful of episodes (real risk
    here: peckingorder has only 63 lost episodes total, each reused ~4x across 237 training pairs
    when pairs_per_game=None)."""
    rng = random.Random(seed)
    train_by_game: dict[str, dict[str, list[dict]]] = {}
    val_by_game: dict[str, dict[str, list[dict]]] = {}
    for game, buckets in by_game.items():
        train_by_game[game] = {}
        val_by_game[game] = {}
        for outcome, episodes in buckets.items():
            episodes = sorted(episodes, key=lambda e: e["episode_id"])
            rng.shuffle(episodes)
            n_val = max(1, round(len(episodes) * val_fraction)) if len(episodes) >= 2 else 0
            val_by_game[game][outcome] = episodes[:n_val]
            train_by_game[game][outcome] = episodes[n_val:]
    return train_by_game, val_by_game


def make_pairs(
    by_game: dict[str, dict[str, list[dict]]], pairs_per_game: int | None = None, seed: int = 42,
) -> list[dict]:
    """random-pair one won episode with one lost episode, within the same game. with
    replacement (sampled independently each draw) since won/lost counts are imbalanced per game
    (e.g. peckingorder: 237 won vs 63 lost) and we want the smaller class to appear more than
    once rather than capping every game at its lost-episode count.

    pairs_per_game=None defaults to max(len(won), len(lost)) per game, so each game contributes a
    comparable number of pairs to the dataset regardless of its win-rate skew."""
    rng = random.Random(seed)
    pairs = []
    for game, buckets in by_game.items():
        won, lost = buckets["won"], buckets["lost"]
        if not won or not lost:
            continue  # need at least one of each outcome to form any pair
        n = pairs_per_game or max(len(won), len(lost))
        for _ in range(n):
            pairs.append({
                "game": game,
                "won_episode": rng.choice(won),
                "lost_episode": rng.choice(lost),
            })
    return pairs


class EpisodePairDataset(Dataset):
    """each item is a (won_text, lost_text) pair, tokenized independently (separate sequences,
    not concatenated -- the preference loss in train.py compares their two summed scores)."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 1024):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        won_enc = self.tokenizer(
            pair["won_episode"]["text"], truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        lost_enc = self.tokenizer(
            pair["lost_episode"]["text"], truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        return {
            "won_input_ids": won_enc["input_ids"].squeeze(0),
            "won_attention_mask": won_enc["attention_mask"].squeeze(0),
            "lost_input_ids": lost_enc["input_ids"].squeeze(0),
            "lost_attention_mask": lost_enc["attention_mask"].squeeze(0),
            "game": pair["game"],
            "won_episode_id": pair["won_episode"]["episode_id"],
            "lost_episode_id": pair["lost_episode"]["episode_id"],
        }


def _pad_stack(seqs: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """pads won and lost sequences SEPARATELY (they're independent episodes of possibly very
    different lengths, not a single padded pair)."""
    return {
        "won_input_ids": _pad_stack([b["won_input_ids"] for b in batch], pad_token_id),
        "won_attention_mask": _pad_stack([b["won_attention_mask"] for b in batch], 0),
        "lost_input_ids": _pad_stack([b["lost_input_ids"] for b in batch], pad_token_id),
        "lost_attention_mask": _pad_stack([b["lost_attention_mask"] for b in batch], 0),
        "games": [b["game"] for b in batch],
    }
