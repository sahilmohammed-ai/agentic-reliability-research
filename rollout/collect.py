"""
collect n rollout trajectories from an agentic environment and save them to disk as json.

zero-coordination baseline: thinker plans once, worker acts until done or the step cap. no
replanning/masking/verifier-driven coordination (see rollout/runner.py's module docstring).

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

from rollout.runner import run_episode, run_single_agent_episode, run_coordinated_episode, DEFAULT_MODEL

# path to base alfworld config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")


def _make_env(env_name: str, split: str, twx_games: tuple[str, ...] | None = None):
    """construct the requested environment wrapper. imports are local so a run that only
    uses one env doesn't pay the other's import cost (alfworld and scienceworld are both heavy).

    twx_games overrides textworldexpress's default game set (coin/simonsays/peckingorder), e.g.
    to test the previously-3b-excluded harder games (cookingworld/twc/mapreader/arithmetic) with
    a stronger worker model. ignored for other envs."""
    if env_name == "alfworld":
        from envs.alfworld_env import AlfWorldEnv
        return AlfWorldEnv(CONFIG_PATH, split=split)
    elif env_name == "scienceworld":
        from envs.scienceworld_env import ScienceWorldEnvWrapper
        return ScienceWorldEnvWrapper(split=split)
    elif env_name == "textworldexpress":
        from envs.textworldexpress_env import TextWorldExpressEnvWrapper, DEFAULT_GAMES
        return TextWorldExpressEnvWrapper(split=split, games=twx_games or DEFAULT_GAMES)
    raise ValueError(f"unknown env: {env_name}")


def collect(
    n: int,
    out_dir: str,
    split: str = "train",
    model: str = DEFAULT_MODEL,
    env_name: str = "alfworld",
    twx_games: tuple[str, ...] | None = None,
    single_agent: bool = False,
    coordinated: bool = False,
) -> None:
    """collect n rollout trajectories and save as json files.

    single_agent=True uses run_single_agent_episode (one model, no thinker plan). coordinated=True
    uses run_coordinated_episode (thinker+worker plus a fixed repetition-triggered replan).
    default (both False) is the zero-coordination thinker+worker loop (run_episode). single_agent
    and coordinated are mutually exclusive."""
    # create output directory and initialize environment
    os.makedirs(out_dir, exist_ok=True)
    env = _make_env(env_name, split, twx_games)
    if single_agent:
        episode_fn = run_single_agent_episode
    elif coordinated:
        episode_fn = run_coordinated_episode
    else:
        episode_fn = run_episode

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
        "--env", type=str, default="alfworld", choices=["alfworld", "scienceworld", "textworldexpress"],
        help="which agentic environment to collect from.",
    )
    parser.add_argument(
        "--twx-games", type=str, default=None,
        help="comma-separated textworldexpress game names to override the default "
             "coin/simonsays/peckingorder set, e.g. --twx-games cookingworld,twc,mapreader,arithmetic. "
             "ignored for other envs.",
    )
    parser.add_argument(
        "--single-agent", action="store_true",
        help="use the single-model baseline (one call per step, no thinker plan) instead of the "
             "default thinker+worker agentic loop.",
    )
    parser.add_argument(
        "--coordinated", action="store_true",
        help="use the thinker+worker loop plus a fixed coordinator: replan when the worker "
             "repeats the same action 3 times in a row. mutually exclusive with --single-agent.",
    )
    args = parser.parse_args()
    if args.single_agent and args.coordinated:
        parser.error("--single-agent and --coordinated are mutually exclusive")
    twx_games = tuple(args.twx_games.split(",")) if args.twx_games else None
    collect(args.n, args.out, args.split, args.model, args.env, twx_games, args.single_agent, args.coordinated)
