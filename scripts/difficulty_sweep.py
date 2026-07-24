"""
difficulty sweep: for each (game, gameParams) candidate, run a small batch of Qwen2.5-3B episodes
and report the win rate. goal is to find configs that land in the ~30-60% win zone -- games where
BOTH success and failure are well-represented, so the verifier has real, varied win/loss contrast
to learn from and be evaluated on.

motivation: the verifier's weakness (build v1-v4) traced to failure diversity. simonsays (~98% win)
and peckingorder (~85%) are too easy for Qwen2.5-3B, so they produce almost no failure examples --
the verifier never learns what failure looks like there and can't be measured there. mapreader
(~30-40% win) is the one game where the verifier actually discriminates well (AUC ~0.78), precisely
because it has balanced outcomes. this sweep tunes the too-easy games' difficulty params to push
them into that balanced zone.

usage (run on a GPU studio -- this is Qwen inference):
    python -m scripts.difficulty_sweep --n 10 --model hf:Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import time

from dotenv import load_dotenv
load_dotenv()

from envs.textworldexpress_env import TextWorldExpressEnvWrapper
from rollout.runner import run_episode

# candidate (game, gameParams) configs to sweep. params tested valid via direct env probing:
# coin numLocations maxes at 11; simonsays gameLength accepts 15/25; mapreader takes numLocations
# + maxDistanceApart. current (too-easy) baselines included for reference.
SWEEP = [
    # coin: harder = more locations + distractors to search through
    ("coin", "numLocations=5,numDistractorItems=3"),    # current baseline
    ("coin", "numLocations=8,numDistractorItems=5"),
    ("coin", "numLocations=11,numDistractorItems=8"),   # hardest valid
    # simonsays: harder = longer instruction sequence (more chances to slip)
    ("simonsays", "gameLength=5"),                        # current baseline
    ("simonsays", "gameLength=10"),
    ("simonsays", "gameLength=15"),
    ("simonsays", "gameLength=25"),
    # peckingorder: no known difficulty param, baseline only (for reference)
    ("peckingorder", ""),
]


def sweep(n: int, model: str, split: str = "train") -> None:
    print(f"difficulty sweep: {n} episodes/config, model={model}, split={split}\n", flush=True)
    results = []
    for game, params in SWEEP:
        # build a single-game env with the custom params by temporarily overriding _GAME_PARAMS
        import envs.textworldexpress_env as twe
        twe._GAME_PARAMS[game] = params
        env = TextWorldExpressEnvWrapper(split=split, games=(game,))

        won = 0
        t0 = time.time()
        for i in range(n):
            traj = run_episode(env, task_id=f"{game}_{i}", model=model)
            won += int(traj.won)
        env.close()

        rate = won / n
        results.append((game, params, rate))
        marker = "  <-- balanced (~30-60%)" if 0.3 <= rate <= 0.6 else ""
        print(f"{game:12} {params:40} | {won}/{n} won ({rate:.0%}) [{time.time()-t0:.0f}s]{marker}", flush=True)

    print("\n=== summary (balanced configs marked) ===", flush=True)
    for game, params, rate in results:
        marker = "  <-- BALANCED" if 0.3 <= rate <= 0.6 else ""
        print(f"  {game:12} {params:40} {rate:.0%}{marker}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="episodes per config")
    parser.add_argument("--model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    sweep(args.n, args.model, args.split)
