"""
prints one mapreader episode's first 1-2 turns cleanly to the terminal, for a paper figure
screenshot. reuses the same env wrapper, thinker, and worker code the rest of the project runs on
(envs/textworldexpress_env.py, agents/thinker.py, agents/worker.py) -- not a reimplementation, so
the figure reflects the actual pipeline.

usage:
    python -m scripts.print_example_turn --game mapreader --n-turns 2
"""

import argparse

from dotenv import load_dotenv
load_dotenv()

from agents import thinker, worker
from envs.textworldexpress_env import TextWorldExpressEnvWrapper


def main(game: str, n_turns: int, worker_model: str, seed: int) -> None:
    env = TextWorldExpressEnvWrapper(split="train", games=(game,))
    obs, info = env.reset_to(game, seed) if hasattr(env, "reset_to") else env.reset()
    task_goal = env.task_goal(obs, info)

    print("=" * 70)
    print(f"GAME: {game}")
    print("=" * 70)
    print(f"\nTASK GOAL:\n{task_goal}\n")

    plan, _ = thinker.plan(task_goal, obs, model=worker_model)
    print(f"PLAN:\n{plan}\n")

    action_history: list[str] = []
    current_obs = obs
    for turn_i in range(1, n_turns + 1):
        admissible = env.admissible_commands(info)
        chosen, _usage = worker.act(
            task_goal, plan, current_obs, admissible, action_history, model=worker_model,
        )
        next_obs, reward, done, info = env.step(chosen)
        action_history.append(chosen)

        print("-" * 70)
        print(f"TURN {turn_i}")
        print("-" * 70)
        print(f"Observation before:\n{current_obs}\n")
        print(f"Admissible commands: {admissible}\n")
        print(f"Action taken: {chosen}\n")
        print(f"Observation after:\n{next_obs}\n")
        print(f"env_reward: {reward}   done: {done}")

        current_obs = next_obs
        if done:
            print("\n(episode ended)")
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="mapreader",
                         choices=["mapreader", "simonsays", "coin", "peckingorder"])
    parser.add_argument("--n-turns", type=int, default=2)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args.game, args.n_turns, args.worker_model, args.seed)
