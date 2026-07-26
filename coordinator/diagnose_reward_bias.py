"""
diagnose whether the coordinator's PPO reward (verifier_v5's advantage on the resulting turn) is
STRUCTURALLY biased toward "retry"-style actions, independent of whether retrying was actually the
right call -- the root-cause hypothesis an independent (Opus advisor) review raised after
coordinator_v1's training collapsed to ~85-92% "retry" in BOTH won and lost episodes.

mechanism under test: run_learned_coordinated_episode's "retry" action masks the worker's actual
last action out of its choice list and adds a "reconsider, pick something different, more
promising" prompt (agents/worker.py's env_hint), then scores whatever action results with
verifier_v5. task_goal/plan/obs_before are IDENTICAL between the "continue" and "retry" paths on
the same turn -- only the action string differs. if the verifier systematically scores the
retry-produced action higher than the actual recorded (continue-path) action, REGARDLESS of
whether the episode ultimately won or lost, that confirms the reward rewards "retry's nudge
produces a locally better-looking action" rather than "was intervening actually the right call" --
a reward-hacking optimum entropy/KL tuning alone cannot fix.

v2 (faithful): the first version of this script used a uniformly RANDOM alternative admissible
command as a proxy for "retry" and found no bias (retry_win_rate ~47%, a coin flip) -- but that
doesn't test the real mechanism, since actual "retry" re-invokes the WORKER LLM with a directed
"reconsider" prompt, not a random swap. this version calls agents.worker.act() for real, exactly
replicating rollout/runner.py's retry branch (mask the real last action out of admissible, append
the same retry_hint string to env_hint), so the alternative action is what the real coordinator's
"retry" would actually produce. needs live LLM calls -- use hf:Qwen/Qwen2.5-3B-Instruct per the
project's local-cost-discipline rule, NOT a frontier model, and NOT Lightning AI (this is cheap
enough to run locally: one extra worker call per sampled turn, no training).

usage:
    python -m coordinator.diagnose_reward_bias --checkpoint checkpoints/verifier_v5 \\
        --rollout-dir data/rollouts/v5_eval_ood --n-per-game 30 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from agents import worker
from verifier.infer import Verifier

GAMES = ("coin", "simonsays", "peckingorder", "mapreader")
RETRY_HINT = (
    "Your previous action did not clearly advance the task. Reconsider the current "
    "observation and choose a different, more promising command."
)


def diagnose(checkpoint: str, rollout_dir: str, n_per_game: int, worker_model: str, seed: int = 42) -> None:
    verifier = Verifier(checkpoint, bound_q_value=False)
    rng = random.Random(seed)

    # {outcome: [(continue_score, retry_score), ...]}
    results = defaultdict(list)

    for game in GAMES:
        files = sorted(glob.glob(os.path.join(rollout_dir, game, "*.json")))
        rng.shuffle(files)
        for path in files[:n_per_game]:
            traj = json.load(open(path))
            outcome = "won" if traj["won"] else "lost"
            worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
            if len(worker_turns) < 2:
                continue
            # skip turn 1 (no coordinator decision precedes it in the real runner) and pick one
            # interior turn per episode, matching the scale of a real per-turn coordinator decision
            turn_idx = rng.randrange(1, len(worker_turns))
            turn = worker_turns[turn_idx]
            admissible = turn["metadata"].get("admissible_commands", [])
            real_action = turn["action"]
            if len(admissible) < 2:
                continue  # retry needs at least one real alternative to mask toward

            # faithfully replicate rollout/runner.py's retry branch: mask the real action out of
            # admissible, append RETRY_HINT to env_hint, re-invoke the SAME worker model with the
            # REAL action history up to this turn (not synthetic)
            action_history = [t["action"] for t in worker_turns[:turn_idx]]
            worker_choices = [c for c in admissible if c != real_action]
            retry_action, _ = worker.act(
                traj["task_goal"], traj["plan"], turn["obs_before"], worker_choices, action_history,
                model=worker_model, env_hint=RETRY_HINT,
            )

            continue_score = verifier.score(traj["task_goal"], traj["plan"], turn["obs_before"], real_action)[1]
            retry_score = verifier.score(traj["task_goal"], traj["plan"], turn["obs_before"], retry_action)[1]
            results[outcome].append((continue_score, retry_score))
            print(f"  [{game}/{outcome}] continue={real_action!r} ({continue_score:.4f}) vs "
                  f"retry={retry_action!r} ({retry_score:.4f})", flush=True)

    print(f"{'outcome':8} {'n':>5} {'continue_mean':>14} {'retry_mean':>12} {'retry_wins':>11} {'retry_win_rate':>15}")
    for outcome in ("won", "lost"):
        pairs = results[outcome]
        if not pairs:
            continue
        n = len(pairs)
        continue_mean = sum(c for c, r in pairs) / n
        retry_mean = sum(r for c, r in pairs) / n
        retry_wins = sum(1 for c, r in pairs if r > c)
        print(f"{outcome:8} {n:>5} {continue_mean:>14.4f} {retry_mean:>12.4f} {retry_wins:>11} {retry_wins/n*100:>14.1f}%")

    all_pairs = results["won"] + results["lost"]
    if all_pairs:
        n = len(all_pairs)
        continue_mean = sum(c for c, r in all_pairs) / n
        retry_mean = sum(r for c, r in all_pairs) / n
        retry_wins = sum(1 for c, r in all_pairs if r > c)
        print(f"\nOVERALL: n={n}, continue_mean={continue_mean:.4f}, retry_mean={retry_mean:.4f}, "
              f"retry scored higher {retry_wins}/{n} ({retry_wins/n*100:.1f}%) of the time")
        print(
            "\nif retry_win_rate is well above 50% in BOTH won and lost episodes (not just lost), "
            "that confirms a structural bias: a masked/alternative action scores better than the "
            "worker's actual choice regardless of whether the episode was succeeding, which is "
            "exactly the reward-hacking pattern that would make 'retry' look attractive to PPO "
            "independent of whether intervening was ever the right call."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/verifier_v5")
    parser.add_argument("--rollout-dir", type=str, default="data/rollouts/v5_eval_ood")
    parser.add_argument("--n-per-game", type=int, default=30)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.rollout_dir, args.n_per_game, args.worker_model, args.seed)
