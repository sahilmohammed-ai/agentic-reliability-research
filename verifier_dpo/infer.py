"""
per-turn score readout + the sanity-check variance report.

reads a trained PreferenceScorer checkpoint, encodes each episode ONCE as a whole (the same shape
training saw, via dataset.py's build_episode_text), and slices each turn's own char span out of
that single encoding using the tokenizer's offset mapping, pooling that turn's own token scores.

fixed 2026-07-29 (review finding): this previously re-encoded each turn in ISOLATION via a
separate build_prefix_text() function with different separators and the plan re-attached to every
turn -- text in a format the model never saw during training (training only ever sees the whole
episode in dataset.py's build_episode_text() shape). that mismatch meant every per-turn score was
being read off an off-distribution input, which could produce a misleading variance report even if
the model itself trained correctly. slicing spans from one real whole-episode encoding removes the
mismatch entirely: the exact tokens fed to the model here are a subsequence of what it was trained
on, not a re-tokenized rebuild of similar-looking text.

this directly answers the one question this sanity check exists to answer: does this training
objective produce real per-step variance within an episode, or does it collapse flat like
label_td.py's TD/GAE relabeling did ("a wash", build 11)? see train.py's module docstring for what
a pass here does and does not prove. per NOTE in run_sanity_report(), turn-length is reported
alongside score so a real variance finding can be distinguished from a length confound (a second
review finding: since won/lost episodes differ systematically in length, per-turn score variance
that's actually just tracking turn token-count would be a false positive for "real per-step
signal")."""

import argparse
import glob
import json
import os
import random

import torch
from transformers import AutoTokenizer

from verifier_dpo.dataset import build_episode_text
from verifier_dpo.model import BASE_MODEL, PreferenceScorer


@torch.no_grad()
def score_episode_turns(
    model, tokenizer, traj: dict, device: str, max_length: int = 2048,
) -> list[tuple[float, int]]:
    """returns one (score, n_tokens) pair per worker turn: the mean per-token score over that
    turn's own char span within a SINGLE whole-episode encoding (not a separate per-turn
    encoding -- see module docstring), plus that span's token count so callers can check for a
    length confound before trusting any variance number."""
    text, turn_spans = build_episode_text(traj)
    enc = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt", return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()  # [(char_start, char_end), ...] per token
    enc = {k: v.to(device) for k, v in enc.items()}

    per_token = model(enc["input_ids"], enc["attention_mask"])[0]  # (seq_len,)

    results: list[tuple[float, int]] = []
    for char_start, char_end in turn_spans:
        # token indices whose char span overlaps this turn's char span. offsets are (0, 0) for
        # special/pad tokens (padding not present here, single unpadded sequence), which naturally
        # fail this overlap test and are excluded.
        token_idxs = [
            i for i, (s, e) in enumerate(offsets)
            if e > char_start and s < char_end and not (s == 0 and e == 0)
        ]
        if not token_idxs:
            # truncated away (episode longer than max_length) -- skip rather than silently
            # recording a bogus score for a turn the model never actually saw.
            continue
        idx_tensor = torch.tensor(token_idxs, device=device)
        turn_score = per_token[idx_tensor].mean().item()
        results.append((turn_score, len(token_idxs)))
    return results


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
    all_scores_lengths: list[tuple[float, int]] = []  # for the length-confound check below

    for path in paths:
        with open(path) as f:
            traj = json.load(f)
        turn_results = score_episode_turns(model, tokenizer, traj, device)
        if not turn_results:
            continue
        scores = [s for s, _ in turn_results]
        lengths = [n for _, n in turn_results]
        all_scores_lengths.extend(turn_results)

        within_ep_std = torch.tensor(scores).std().item() if len(scores) > 1 else 0.0
        report["episodes"].append({
            "episode_id": os.path.basename(path),
            "won": traj["won"],
            "n_turns": len(scores),
            "scores": [round(s, 4) for s in scores],
            "turn_token_lengths": lengths,
            "within_episode_std": round(within_ep_std, 4),
        })

        # opportunistically collect same-(game, obs_before) turns across different episodes/
        # actions, to directly check same-state discrimination too (not the primary claim of this
        # method, but cheap to check on the same pass and directly relevant to the standing
        # question this whole investigation is about).
        worker_turns = [t for t in traj["turns"] if t["role"] == "worker"][: len(scores)]
        for turn, score in zip(worker_turns, scores):
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

    # length-confound check (review finding): if score is really just tracking turn token-count,
    # the correlation below will be strongly nonzero, and within_episode_std must then be read as
    # "how much turn length varies," not "how much the model actually discriminated turn quality."
    length_corr = 0.0
    if len(all_scores_lengths) > 2:
        s_t = torch.tensor([s for s, _ in all_scores_lengths])
        l_t = torch.tensor([float(n) for _, n in all_scores_lengths])
        if s_t.std() > 0 and l_t.std() > 0:
            length_corr = torch.corrcoef(torch.stack([s_t, l_t]))[0, 1].item()

    report["summary"] = {
        "n_episodes": len(report["episodes"]),
        "mean_within_episode_std": round(sum(all_stds) / len(all_stds), 4) if all_stds else 0.0,
        "n_same_state_groups_found": len(report["same_state_pairs"]),
        "score_vs_turn_length_correlation": round(length_corr, 4),
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
