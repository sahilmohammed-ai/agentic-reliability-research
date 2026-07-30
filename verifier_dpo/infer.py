"""
per-turn score readout + the sanity-check variance report.

reads a trained PreferenceScorer checkpoint, re-encodes each episode turn-by-turn (one forward
pass per prefix, simplest correct approach -- no need for exact token-span slicing of the
whole-episode encoding), and pools each turn's own action-span score. this directly answers the
one question this sanity check exists to answer: does this training objective produce real
per-step variance within an episode, or does it collapse flat like label_td.py's TD/GAE relabeling
did ("a wash", build 11)? see train.py's module docstring for what a pass here does and does not
prove.
"""

import argparse
import glob
import json
import os
import random

import torch
from transformers import AutoTokenizer

from verifier_dpo.model import BASE_MODEL, PreferenceScorer


def build_prefix_text(task_goal: str, plan: str, obs_before: str, action: str) -> str:
    """same per-turn shape as verifier/dataset.py's build_input_text, so scores are comparable
    to the existing verifier's per-turn granularity."""
    return (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Observation:\n{obs_before}\n\n"
        f"Action taken: {action}"
    )


@torch.no_grad()
def score_episode_turns(model, tokenizer, traj: dict, device: str, batch_size: int = 8) -> list[float]:
    """returns one scalar per worker turn: the mean per-token score over that turn's own
    (task/plan/obs/action) encoding -- the same per-turn text unit the existing verifier scores,
    so this is a like-for-like comparison, not a different granularity.

    batches all of an episode's turns together (single padded forward pass per batch) instead of
    one forward pass per turn -- on CPU, per-turn calls were the actual bottleneck (confirmed: a
    6-episode, ~15-turns/episode run took 25+ min of CPU time with nothing returned), not model
    size alone. batching is a straightforward speedup with no change in what's being measured."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    if not worker_turns:
        return []

    texts = [
        build_prefix_text(traj["task_goal"], traj["plan"], t["obs_before"], t["action"])
        for t in worker_turns
    ]
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=512, padding=True, return_tensors="pt",
        ).to(device)
        per_token = model(enc["input_ids"], enc["attention_mask"])  # (batch, seq_len)
        mask = enc["attention_mask"].float()
        turn_scores = (per_token * mask).sum(dim=1) / mask.sum(dim=1)
        scores.extend(turn_scores.tolist())
    return scores


def run_sanity_report(checkpoint: str, labeled_dir: str, n_episodes: int, seed: int = 42) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PreferenceScorer(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    paths = sorted(glob.glob(os.path.join(labeled_dir, "*.json")))
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:n_episodes]

    report = {"episodes": [], "same_state_pairs": []}
    obs_to_scores: dict[tuple[str, str], list[float]] = {}

    for path in paths:
        with open(path) as f:
            traj = json.load(f)
        scores = score_episode_turns(model, tokenizer, traj, device)
        if not scores:
            continue

        within_ep_std = torch.tensor(scores).std().item() if len(scores) > 1 else 0.0
        report["episodes"].append({
            "episode_id": os.path.basename(path),
            "won": traj["won"],
            "n_turns": len(scores),
            "scores": [round(s, 4) for s in scores],
            "within_episode_std": round(within_ep_std, 4),
        })

        # opportunistically collect same-(game, obs_before) turns across different episodes/
        # actions, to directly check same-state discrimination too (not the primary claim of this
        # method, but cheap to check on the same pass and directly relevant to the standing
        # question this whole investigation is about).
        for turn, score in zip([t for t in traj["turns"] if t["role"] == "worker"], scores):
            key = (traj.get("task_goal", ""), turn["obs_before"])
            obs_to_scores.setdefault(key, []).append((turn["action"], score))

    for key, action_scores in obs_to_scores.items():
        distinct_actions = {a for a, _ in action_scores}
        if len(distinct_actions) > 1 and len(action_scores) > 1:
            vals = [s for _, s in action_scores]
            spread = max(vals) - min(vals)
            report["same_state_pairs"].append({
                "obs_before": key[1][:120],
                "actions_and_scores": action_scores,
                "spread": round(spread, 4),
            })

    all_stds = [e["within_episode_std"] for e in report["episodes"]]
    report["summary"] = {
        "n_episodes": len(report["episodes"]),
        "mean_within_episode_std": round(sum(all_stds) / len(all_stds), 4) if all_stds else 0.0,
        "n_same_state_groups_found": len(report["same_state_pairs"]),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="verifier_dpo/checkpoints/sanity_check/model.pt")
    parser.add_argument("--labeled-dir", type=str, default="data/labeled/v5_train_combined")
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--out", type=str, default="verifier_dpo/checkpoints/sanity_check/report.json")
    args = parser.parse_args()

    report = run_sanity_report(args.checkpoint, args.labeled_dir, args.n_episodes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))
    print(f"\nfull report -> {args.out}")
