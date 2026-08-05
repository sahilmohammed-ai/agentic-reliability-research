"""
independent, held-out validation of verifier/frozen_llm.py's score_candidate() -- the pre-hoc
judge added for Best-of-N (scripts/best_of_n_eval.py's --scorer llm), which had no standalone
evaluation before being used there (unlike verifier_mc's AUC or verifier_dpo's episode accuracy,
both checked BEFORE being used in Best-of-N).

scores every worker turn in the real eval_ood rollouts (data/rollouts/v5_eval_ood/, the SAME
episodes verifier_mc's AUC was computed on) using score_candidate() -- note this scores the turn's
ACTUAL obs_before/action pair, same real turns, just under the pre-hoc (no obs_after) judging task
rather than build 08/09's post-hoc score_turn(). reports won/lost mean-score separation and AUC,
directly comparable to verifier_mc's per-game AUC table and build 08's won/lost mean-score numbers.

usage:
    python -m scripts.score_prehoc_llm_verifier --max-episodes-per-game 15 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/prehoc_llm_scores.json
"""

import argparse
import glob
import json
import os

from dotenv import load_dotenv
load_dotenv()

from verifier.frozen_llm import score_candidate


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """plain rank-based AUC (Mann-Whitney U / probability a random positive outscores a random
    negative), no sklearn dependency needed. returns None if only one class is present."""
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


def score_eval_ood(
    rollouts_dir: str, max_episodes_per_game: int | None, worker_model: str,
) -> dict:
    games = sorted(os.listdir(rollouts_dir))
    per_game_turn_scores: dict[str, list[tuple[float, int]]] = {}  # game -> [(score, won_label)]

    for game in games:
        game_dir = os.path.join(rollouts_dir, game)
        if not os.path.isdir(game_dir):
            continue
        paths = sorted(glob.glob(os.path.join(game_dir, "*.json")))
        if max_episodes_per_game is not None:
            paths = paths[:max_episodes_per_game]

        scores_labels: list[tuple[float, int]] = []
        for i, path in enumerate(paths):
            with open(path) as f:
                ep = json.load(f)
            won_label = int(bool(ep.get("won")))
            worker_turns = [t for t in ep["turns"] if t["role"] == "worker"]
            action_history: list[str] = []
            for turn in worker_turns:
                score, _usage = score_candidate(
                    ep["task_goal"], action_history, turn["obs_before"], turn["action"],
                    model=worker_model,
                )
                scores_labels.append((score, won_label))
                action_history.append(turn["action"])
            print(f"  [{game} {i+1}/{len(paths)}] won={bool(won_label)} "
                  f"turns_scored={len(worker_turns)}", flush=True)

        per_game_turn_scores[game] = scores_labels

    results = {"games": {}}
    for game, scores_labels in per_game_turn_scores.items():
        if not scores_labels:
            continue
        scores = [s for s, _ in scores_labels]
        labels = [y for _, y in scores_labels]
        won_scores = [s for s, y in scores_labels if y == 1]
        lost_scores = [s for s, y in scores_labels if y == 0]
        auc = _roc_auc(scores, labels)
        results["games"][game] = {
            "won_mean": sum(won_scores) / len(won_scores) if won_scores else None,
            "lost_mean": sum(lost_scores) / len(lost_scores) if lost_scores else None,
            "auc": auc,
            "won_n": len(won_scores),
            "lost_n": len(lost_scores),
        }
        print(f"{game}: won_mean={results['games'][game]['won_mean']} "
              f"lost_mean={results['games'][game]['lost_mean']} auc={auc} "
              f"(won_n={len(won_scores)}, lost_n={len(lost_scores)})")

    all_scores_labels = [sl for g in per_game_turn_scores.values() for sl in g]
    if all_scores_labels:
        all_scores = [s for s, _ in all_scores_labels]
        all_labels = [y for _, y in all_scores_labels]
        won_all = [s for s, y in all_scores_labels if y == 1]
        lost_all = [s for s, y in all_scores_labels if y == 0]
        results["overall"] = {
            "won_mean": sum(won_all) / len(won_all) if won_all else None,
            "lost_mean": sum(lost_all) / len(lost_all) if lost_all else None,
            "auc": _roc_auc(all_scores, all_labels),
            "won_n": len(won_all),
            "lost_n": len(lost_all),
        }
        print(f"OVERALL (pooled): {results['overall']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-dir", type=str, default="data/rollouts/v5_eval_ood")
    parser.add_argument("--max-episodes-per-game", type=int, default=None,
                         help="cap episodes/game to keep cost down (each turn = 1 real LLM call). "
                              "default None scores all 75/game (300 total, ~6500 turns).")
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", type=str, default="reports/prehoc_llm_scores.json")
    args = parser.parse_args()

    results = score_eval_ood(args.rollouts_dir, args.max_episodes_per_game, args.worker_model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results -> {args.out}")
