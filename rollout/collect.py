"""
collect n rollout trajectories from TextWorldExpress and save them to disk as json.

zero-coordination baseline: thinker plans once, worker acts until done or the step cap. no
replanning/masking/verifier-driven coordination (see rollout/runner.py's module docstring).

usage:
    python -m rollout.collect --n 10 --out data/rollouts/build_9 --model hf:Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from envs.textworldexpress_env import TextWorldExpressEnvWrapper, DEFAULT_GAMES
from rollout.runner import run_episode, run_single_agent_episode, DEFAULT_MODEL


def collect(
    n: int,
    out_dir: str,
    split: str = "train",
    model: str = DEFAULT_MODEL,
    twx_games: tuple[str, ...] | None = None,
    single_agent: bool = False,
) -> None:
    """collect n rollout trajectories and save as json files.

    single_agent=True uses run_single_agent_episode (one model, no thinker plan). default (False)
    is the zero-coordination thinker+worker loop (run_episode)."""
    # create output directory and initialize environment
    os.makedirs(out_dir, exist_ok=True)
    env = TextWorldExpressEnvWrapper(split=split, games=twx_games or DEFAULT_GAMES)
    episode_fn = run_single_agent_episode if single_agent else run_episode

    won_count = 0
    failed = 0
    for i in range(n):
        # run one episode and record trajectory
        task_id = f"{split}_{i:04d}"
        t0 = time.time()
        try:
            traj = episode_fn(env, task_id=task_id, model=model)
        except ValueError as e:
            # confirmed cause: claude-sonnet-5 occasionally returns only a ThinkingBlock with no
            # text, even after llm.py's internal retries. skip this episode rather than crash the
            # whole collection run.
            failed += 1
            print(f"[{i+1}/{n}] {task_id} | SKIPPED after error: {e}")
            continue
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
    collected = n - failed
    print(f"\nDone. {won_count}/{collected} episodes won ({failed} skipped due to errors). Trajectories saved to {out_dir}/")


if __name__ == "__main__":
    # parse cli arguments and collect rollouts
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--out", type=str, default="data/rollouts/train")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval_id", "eval_ood"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="model used for both thinker and worker")
    parser.add_argument(
        "--twx-games", type=str, default=None,
        help="comma-separated textworldexpress game names to override the default "
             "coin/simonsays/peckingorder set, e.g. --twx-games cookingworld,twc,mapreader,arithmetic.",
    )
    parser.add_argument(
        "--single-agent", action="store_true",
        help="use the single-model baseline (one call per step, no thinker plan) instead of the "
             "default thinker+worker agentic loop.",
    )
    args = parser.parse_args()
    twx_games = tuple(args.twx_games.split(",")) if args.twx_games else None
    collect(
        args.n, args.out, args.split, args.model, twx_games, args.single_agent,
    )
