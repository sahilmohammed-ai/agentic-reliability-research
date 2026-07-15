"""
collect n rollout trajectories from an agentic environment and save them to disk as json.

usage:
    python -m rollout.collect --n 10 --out data/rollouts/build_2 --model claude-opus-4-8
    python -m rollout.collect --n 10 --out data/rollouts/build_9 --env scienceworld --model hf:Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from rollout.runner import run_episode, DEFAULT_MODEL

# path to base alfworld config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")


def _make_env(env_name: str, split: str):
    """construct the requested environment wrapper. imports are local so a run that only
    uses one env doesn't pay the other's import cost (alfworld and scienceworld are both heavy)."""
    if env_name == "alfworld":
        from envs.alfworld_env import AlfWorldEnv
        return AlfWorldEnv(CONFIG_PATH, split=split)
    elif env_name == "scienceworld":
        from envs.scienceworld_env import ScienceWorldEnvWrapper
        return ScienceWorldEnvWrapper(split=split)
    elif env_name == "textworldexpress":
        from envs.textworldexpress_env import TextWorldExpressEnvWrapper
        return TextWorldExpressEnvWrapper(split=split)
    raise ValueError(f"unknown env: {env_name}")


def collect(
    n: int,
    out_dir: str,
    split: str = "train",
    model: str = DEFAULT_MODEL,
    replan_mode: str = "interval",
    env_name: str = "alfworld",
    verifier_checkpoint: str | None = None,
) -> None:
    """collect n rollout trajectories and save as json files.

    verifier_checkpoint is required when replan_mode == "verifier" (coordinator v0): a directory
    containing a trained verifier.pt, e.g. "checkpoints/verifier_v2"."""
    # create output directory and initialize environment
    os.makedirs(out_dir, exist_ok=True)
    env = _make_env(env_name, split)

    verifier = None
    if replan_mode == "verifier":
        if not verifier_checkpoint:
            raise ValueError('--replan-mode verifier requires --verifier-checkpoint <dir>')
        from verifier.infer import Verifier
        verifier = Verifier(verifier_checkpoint)

    won_count = 0
    for i in range(n):
        # run one episode and record trajectory
        task_id = f"{split}_{i:04d}"
        t0 = time.time()
        traj = run_episode(env, task_id=task_id, model=model, replan_mode=replan_mode, verifier=verifier)
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
    parser.add_argument(
        "--replan-mode", type=str, default="interval",
        choices=["interval", "stagnation", "mask", "verifier"],
        help="interval: replan every N steps (build_6). stagnation: replan when stuck (build_7). "
             "mask: stagnation replan plus single-turn masking of the most-repeated action (build_8). "
             "verifier: coordinator v0, live-verifier-driven continue/retry/replan/backtrack/escalate.",
    )
    parser.add_argument(
        "--env", type=str, default="alfworld", choices=["alfworld", "scienceworld", "textworldexpress"],
        help="which agentic environment to collect from.",
    )
    parser.add_argument(
        "--verifier-checkpoint", type=str, default=None,
        help="directory with a trained verifier.pt, required for --replan-mode verifier.",
    )
    args = parser.parse_args()
    collect(args.n, args.out, args.split, args.model, args.replan_mode, args.env, args.verifier_checkpoint)
