"""
score every turn in the balanced eval_ood set (build 12) with a trained verifier checkpoint
(verifier/infer.py's Verifier class) and save scores for evaluation, mirroring build 08's frozen
LLM verifier evaluation but with a trained checkpoint instead of a prompted judge.

data/rollouts/v5_eval_ood/ was collected fresh under the difficulty-calibrated _GAME_PARAMS
(same as v5's training data) on the --split eval_ood fold, never trained on by any checkpoint --
a genuine held-out evaluation, not a memorization check. cookingworld is excluded (dropped from
the training mix in build 11).

usage:
    python -m scripts.score_trained_verifier --checkpoint checkpoints/verifier_mc \\
        --out data/labeled/v5_eval_scores.json
"""

import argparse
import glob
import json
import time

from verifier.infer import Verifier

GAMES = ["coin", "simonsays", "peckingorder", "mapreader"]
SRC_DIR = "data/rollouts/v5_eval_ood"


def score_episode(traj: dict, verifier: Verifier) -> list[dict]:
    """score every worker turn in one trajectory. returns a list of per-turn score records."""
    records = []
    for turn in traj["turns"]:
        if turn["role"] != "worker":
            continue
        q_value, advantage = verifier.score(
            traj["task_goal"], traj.get("plan", ""), turn["obs_before"], turn["action"]
        )
        records.append({
            "step": turn["step"],
            "action": turn["action"],
            "env_reward": turn["env_reward"],
            "q_value": q_value,
            "advantage": advantage,
        })
    return records


def main(checkpoint_dir: str, out_path: str) -> None:
    verifier = Verifier(checkpoint_dir, bound_q_value=False)

    all_episodes = []
    for game in GAMES:
        files = sorted(glob.glob(f"{SRC_DIR}/{game}/*.json"))
        for i, f in enumerate(files):
            traj = json.load(open(f))
            t0 = time.time()
            turn_records = score_episode(traj, verifier)
            elapsed = time.time() - t0
            all_episodes.append({
                "game": game,
                "task_id": traj["task_id"],
                "won": traj["won"],
                "total_steps": traj["total_steps"],
                "turns": turn_records,
            })
            print(f"[{game} {i+1}/{len(files)}] won={traj['won']} turns={len(turn_records)} ({elapsed:.1f}s)", flush=True)

    with open(out_path, "w") as f:
        json.dump(all_episodes, f, indent=2)
    print(f"\nDone. Scored {len(all_episodes)} episodes -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/verifier_mc")
    parser.add_argument("--out", type=str, default="data/labeled/v5_eval_scores.json")
    args = parser.parse_args()
    main(args.checkpoint, args.out)
