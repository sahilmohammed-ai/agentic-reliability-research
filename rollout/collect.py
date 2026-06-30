"""
Collect N rollout trajectories from ALFWorld and save them to disk as JSON.

Usage:
    python -m rollout.collect --n 10 --out data/rollouts/train --split train
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from envs.alfworld_env import AlfWorldEnv
from rollout.runner import run_episode

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")


def collect(n: int, out_dir: str, split: str = "train") -> None:
    os.makedirs(out_dir, exist_ok=True)
    env = AlfWorldEnv(CONFIG_PATH, split=split)

    won_count = 0
    for i in range(n):
        task_id = f"{split}_{i:04d}"
        t0 = time.time()
        traj = run_episode(env, task_id=task_id)
        elapsed = time.time() - t0

        path = os.path.join(out_dir, f"{task_id}.json")
        with open(path, "w") as f:
            json.dump(traj.to_dict(), f, indent=2)

        won_count += int(traj.won)
        print(
            f"[{i+1}/{n}] {task_id} | steps={traj.total_steps} won={traj.won} "
            f"({elapsed:.1f}s) -> {path}"
        )

    env.close()
    print(f"\nDone. {won_count}/{n} episodes won. Trajectories saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--out", type=str, default="data/rollouts/train")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval_id", "eval_ood"])
    args = parser.parse_args()
    collect(args.n, args.out, args.split)
