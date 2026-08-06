"""
tests whether verifier_dpo's per-turn scores are actual turn-level signal, using the SAME AUC and
failure-detection methodology already validated for verifier_mc -- not an anecdotal read of one
trace (the mapreader same-state example), a real held-out measurement.

why this wasn't done before: verifier_dpo was trained ONLY on whole-episode comparisons
(episode_score() over an entire trajectory), so nothing guarantees its per-token scores are
meaningful at the turn level -- one real trace (mapreader) showed two opposite actions scoring
nearly identically at the same state. that's real, disclosed evidence of a weakness, but it's a
single anecdote, not a controlled measurement. this script gets the actual number.

mechanism: reuses verifier_dpo/infer.py's score_episode_turns() (encodes each episode ONCE as a
whole, slices real per-turn scores via the tokenizer's offset mapping -- the correct,
train/inference-consistent way to read a per-turn score off this model, already validated and
fixed once before, see that file's docstring) on the SAME real eval_ood episodes
(data/rollouts/v5_eval_ood/) verifier_mc's AUC (0.844 pooled) and failure-detection F1 (0.796
mean) were computed on -- a genuinely fair, same-data, same-protocol comparison.

outputs a scores file in the exact schema scripts/failure_detection_eval.py already consumes
(a list of {game, task_id, won, total_steps, turns: [{step, action, env_reward, q_value}]}), using
the DPO per-turn score AS q_value, so the existing AUC and failure-detection pipelines can be run
against it completely unchanged -- no separate metric-computation logic to maintain or accidentally
make inconsistent with verifier_mc's.

usage:
    python -m scripts.score_dpo_turnlevel \\
        --checkpoint verifier_dpo/checkpoints/finetune_v2/model.pt \\
        --rollouts-dir data/rollouts/v5_eval_ood \\
        --out data/labeled/dpo_eval_scores.json
    # then, same pipelines already used for verifier_mc:
    python -m scripts.compute_auc --scores data/labeled/dpo_eval_scores.json  # if/when built
    python -m scripts.failure_detection_eval --scores data/labeled/dpo_eval_scores.json \\
        --score-field q_value --out reports/failure_detection_dpo.json
"""

import argparse
import glob
import json
import os

import torch
from transformers import AutoTokenizer

from verifier_dpo.infer import score_episode_turns
from verifier_dpo.model import BASE_MODEL, PreferenceScorer


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """same plain rank-based AUC as scripts/score_prehoc_llm_verifier.py, kept duplicated (not
    imported) so this script has no import-time dependency on that one -- both are small, stable,
    and independently useful standalone."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def score_and_save(
    checkpoint: str, rollouts_dir: str, out_path: str, max_episodes_per_game: int | None,
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PreferenceScorer(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    games = sorted(os.listdir(rollouts_dir))
    all_scored_episodes = []
    per_game_turn_scores: dict[str, list[tuple[float, int]]] = {}

    for game in games:
        game_dir = os.path.join(rollouts_dir, game)
        if not os.path.isdir(game_dir):
            continue
        paths = sorted(glob.glob(os.path.join(game_dir, "*.json")))
        if max_episodes_per_game is not None:
            paths = paths[:max_episodes_per_game]

        scores_labels = []
        for i, path in enumerate(paths):
            with open(path) as f:
                traj = json.load(f)
            turn_results = score_episode_turns(model, tokenizer, traj, device)
            if not turn_results:
                continue

            worker_turns = [t for t in traj["turns"] if t["role"] == "worker"][: len(turn_results)]
            scored_turns = []
            for turn, (score, _n_tokens) in zip(worker_turns, turn_results):
                scored_turns.append({
                    "step": turn["step"], "action": turn["action"],
                    "env_reward": turn.get("env_reward", 0.0), "q_value": score,
                })
                scores_labels.append((score, int(bool(traj["won"]))))

            all_scored_episodes.append({
                "game": game, "task_id": traj["task_id"], "won": bool(traj["won"]),
                "total_steps": traj["total_steps"], "turns": scored_turns,
            })
            print(f"  [{game} {i+1}/{len(paths)}] won={bool(traj['won'])} "
                  f"turns_scored={len(scored_turns)}", flush=True)

        per_game_turn_scores[game] = scores_labels

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_scored_episodes, f, indent=2)
    print(f"\nscored episodes -> {out_path}")

    print("\nper-game AUC (verifier_dpo per-turn score, real held-out eval_ood):")
    all_scores_labels = []
    for game, scores_labels in per_game_turn_scores.items():
        if not scores_labels:
            continue
        scores = [s for s, _ in scores_labels]
        labels = [y for _, y in scores_labels]
        won_n = sum(labels)
        lost_n = len(labels) - won_n
        auc = _roc_auc(scores, labels)
        print(f"  {game}: auc={auc} (won_n={won_n}, lost_n={lost_n})")
        all_scores_labels.extend(scores_labels)

    if all_scores_labels:
        all_scores = [s for s, _ in all_scores_labels]
        all_labels = [y for _, y in all_scores_labels]
        overall_auc = _roc_auc(all_scores, all_labels)
        print(f"  OVERALL (pooled): auc={overall_auc} (n={len(all_scores_labels)})")

    return all_scored_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                         default="verifier_dpo/checkpoints/finetune_v2/model.pt")
    parser.add_argument("--rollouts-dir", type=str, default="data/rollouts/v5_eval_ood")
    parser.add_argument("--out", type=str, default="data/labeled/dpo_eval_scores.json")
    parser.add_argument("--max-episodes-per-game", type=int, default=None)
    args = parser.parse_args()
    score_and_save(args.checkpoint, args.rollouts_dir, args.out, args.max_episodes_per_game)
