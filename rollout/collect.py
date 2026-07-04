"""
collect n rollout trajectories from alfworld and save them to disk as json.

usage:
    python -m rollout.collect --n 10 --out data/rollouts/build_2 --model claude-opus-4-8
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from envs.alfworld_env import AlfWorldEnv
from rollout.runner import run_episode, DEFAULT_MODEL

# path to base alfworld config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")


def collect(n: int, out_dir: str, split: str = "train", model: str = DEFAULT_MODEL) -> None:
    """collect n rollout trajectories and save as json files."""
    # create output directory and initialize environment
    os.makedirs(out_dir, exist_ok=True)
    env = AlfWorldEnv(CONFIG_PATH, split=split)

    won_count = 0
    for i in range(n):
        # run one episode and record trajectory
        task_id = f"{split}_{i:04d}"
        t0 = time.time()
        traj = run_episode(env, task_id=task_id, model=model)
        elapsed = time.time() - t0

        # save trajectory to json file
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
    # parse cli arguments and collect rollouts
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--out", type=str, default="data/rollouts/train")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval_id", "eval_ood"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="model used for both thinker and worker")
    args = parser.parse_args()
    collect(args.n, args.out, args.split, args.model)
