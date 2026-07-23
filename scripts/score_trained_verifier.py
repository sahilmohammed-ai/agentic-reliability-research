"""
score every turn in build 03's held-out eval_ood baseline with the trained verifier v3 checkpoint
(verifier/infer.py's Verifier class) and save scores for evaluation, mirroring build 08's frozen
LLM verifier evaluation but with a trained checkpoint instead of a prompted judge.

data/rollouts/twx/03_qwen_stratified/ was NEVER used to train verifier v3 (v3 trained on a
separate --split train collection, data/labeled/twx/v3_train_combined/), so this is a genuine
held-out evaluation, not a memorization check.

usage:
    python -m scripts.score_trained_verifier --checkpoint checkpoints/verifier_v3 \\
        --out data/labeled/twx/v3_eval_scores.json
"""

import argparse
import glob
import json
import time

from verifier.infer import Verifier

GAMES = ["coin", "simonsays", "peckingorder", "cookingworld", "mapreader"]
SRC_DIR = "data/rollouts/03_qwen_stratified"


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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/verifier_v3")
    parser.add_argument("--out", type=str, default="data/labeled/twx/v3_eval_scores.json")
    args = parser.parse_args()
    main(args.checkpoint, args.out)
