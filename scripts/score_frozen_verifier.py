"""
score every turn in build 03's baseline trajectories with a frozen LLM verifier and save the
scores alongside episode metadata for evaluation.

no new rollouts -- scores existing data from data/rollouts/twx/03_qwen_stratified/.

usage:
    python -m scripts.score_frozen_verifier --out data/labeled/twx/08_frozen_verifier_scores.json
    python -m scripts.score_frozen_verifier --variant no_reasoning \\
        --out data/labeled/twx/09_frozen_verifier_no_reasoning_scores.json
"""

import argparse
import glob
import importlib
import json
import time

from dotenv import load_dotenv
load_dotenv()

GAMES = ["coin", "simonsays", "peckingorder", "cookingworld", "mapreader"]
SRC_DIR = "data/rollouts/twx/03_qwen_stratified"
MODEL = "hf:Qwen/Qwen2.5-3B-Instruct"

VARIANTS = {
    "reasoning": "verifier.frozen_llm",
    "no_reasoning": "verifier.frozen_llm_no_reasoning",
}


def score_episode(traj: dict, score_turn) -> list[dict]:
    """score every turn in one trajectory. returns a list of per-turn score records."""
    task_goal = traj["task_goal"]
    history: list[str] = []
    records = []
    for turn in traj["turns"]:
        score, usage = score_turn(
            task_goal=task_goal,
            action_history=history,
            obs_before=turn["obs_before"],
            action=turn["action"],
            obs_after=turn["obs_after"],
            model=MODEL,
        )
        records.append({
            "step": turn["step"],
            "action": turn["action"],
            "env_reward": turn["env_reward"],
            "verifier_score": score,
        })
        history.append(turn["action"])
    return records


def main(out_path: str, variant: str) -> None:
    score_turn = importlib.import_module(VARIANTS[variant]).score_turn

    all_episodes = []
    for game in GAMES:
        files = sorted(glob.glob(f"{SRC_DIR}/{game}/*.json"))
        for i, f in enumerate(files):
            traj = json.load(open(f))
            t0 = time.time()
            turn_records = score_episode(traj, score_turn)
            elapsed = time.time() - t0
            all_episodes.append({
                "game": game,
                "task_id": traj["task_id"],
                "won": traj["won"],
                "total_steps": traj["total_steps"],
                "turns": turn_records,
            })
            print(f"[{game} {i+1}/{len(files)}] won={traj['won']} turns={len(turn_records)} ({elapsed:.1f}s)")

    with open(out_path, "w") as f:
        json.dump(all_episodes, f, indent=2)
    print(f"\nDone. Scored {len(all_episodes)} episodes -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/labeled/twx/08_frozen_verifier_scores.json")
    parser.add_argument("--variant", type=str, default="reasoning", choices=list(VARIANTS))
    args = parser.parse_args()
    main(args.out, args.variant)
