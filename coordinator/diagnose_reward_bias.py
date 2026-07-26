"""
two diagnostics, run together, both BEFORE spending Lightning AI time on a third coordinator
training attempt:

(1) reward-bias check: does verifier_v5's advantage STRUCTURALLY favor "retry"-produced actions,
independent of whether retrying was the right call? Tested and RULED OUT (2026-07-26, 115 real
comparisons): retry-produced actions scored LOWER on average than the real recorded action
(-0.1306 vs -0.0222), won only 42.6% of head-to-head comparisons. Kept in this script as a
regression check, not because it's still an open question.

(2) reward-separability check (the load-bearing one after coordinator_v2's postmortem): is the
verifier's advantage even DIFFERENT enough across continue/retry/replan, at the ~20-episode/
iteration scale train_ppo.py actually uses, for PPO to have anything to learn from? coordinator_v2
trained with a policy head that (before its zero-init fix) started heavily biased toward "retry"
and NEVER moved off that bias across 10 iterations (retry-share oscillated flat at 80-85%, no
trend) -- consistent with a reward signal too weak/noisy at this scale to overcome even a
CORRECTED (now-uniform) initialization. If continue/retry/replan's mean rewards are statistically
indistinguishable here, no amount of policy-side fixes (zero-init, entropy, anything) will make
PPO learn real discrimination -- the fix would be more episodes/iteration or a different reward
design, not another coordinator-code change.

mechanism: for each sampled real turn, compute what EACH of the 3 actions would actually produce
and score with verifier_v5. "continue" = the real recorded action (ground truth). "retry" =
worker.act() re-invoked with the real action masked out + the exact RETRY_HINT rollout/runner.py's
retry branch uses. "replan" = thinker.replan() called on the real prior plan/history, then
worker.act() under the NEW plan. All three go through the real worker/thinker LLM calls, not
synthetic substitutes -- needs hf:Qwen/Qwen2.5-3B-Instruct per the project's local-cost-discipline
rule, NOT a frontier model, and NOT Lightning AI (cheap enough locally: 2 extra LLM calls per
sampled turn, no training).

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
import statistics
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from agents import thinker, worker
from verifier.infer import Verifier

GAMES = ("coin", "simonsays", "peckingorder", "mapreader")
RETRY_HINT = (
    "Your previous action did not clearly advance the task. Reconsider the current "
    "observation and choose a different, more promising command."
)


def diagnose(checkpoint: str, rollout_dir: str, n_per_game: int, worker_model: str, seed: int = 42) -> None:
    verifier = Verifier(checkpoint, bound_q_value=False)
    rng = random.Random(seed)

    # {outcome: [{"continue": score, "retry": score, "replan": score}, ...]}
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

            action_history = [t["action"] for t in worker_turns[:turn_idx]]

            # continue: the real recorded action, ground truth
            continue_score = verifier.score(traj["task_goal"], traj["plan"], turn["obs_before"], real_action)[1]

            # retry: faithfully replicate rollout/runner.py's retry branch (mask the real action,
            # append RETRY_HINT, re-invoke the SAME worker model with the real history)
            worker_choices = [c for c in admissible if c != real_action]
            retry_action, _ = worker.act(
                traj["task_goal"], traj["plan"], turn["obs_before"], worker_choices, action_history,
                model=worker_model, env_hint=RETRY_HINT,
            )
            retry_score = verifier.score(traj["task_goal"], traj["plan"], turn["obs_before"], retry_action)[1]

            # replan: faithfully replicate rollout/runner.py's replan branch (thinker.replan() on
            # the real prior plan/history, then worker.act() under the NEW plan, full admissible
            # list -- replan does not mask anything)
            new_plan, _ = thinker.replan(
                traj["task_goal"], traj["plan"], action_history, turn["obs_before"], model=worker_model,
            )
            replan_action, _ = worker.act(
                traj["task_goal"], new_plan, turn["obs_before"], admissible, action_history,
                model=worker_model,
            )
            replan_score = verifier.score(traj["task_goal"], new_plan, turn["obs_before"], replan_action)[1]

            results[outcome].append({"continue": continue_score, "retry": retry_score, "replan": replan_score})
            print(
                f"  [{game}/{outcome}] continue={real_action!r} ({continue_score:.4f}) | "
                f"retry={retry_action!r} ({retry_score:.4f}) | "
                f"replan={replan_action!r} ({replan_score:.4f})", flush=True,
            )

    print(f"\n{'outcome':8} {'n':>5} {'continue_mean':>14} {'retry_mean':>12} {'replan_mean':>13} "
          f"{'retry_wins%':>12} {'replan_wins%':>13}")
    for outcome in ("won", "lost"):
        rows = results[outcome]
        if not rows:
            continue
        n = len(rows)
        continue_mean = sum(r["continue"] for r in rows) / n
        retry_mean = sum(r["retry"] for r in rows) / n
        replan_mean = sum(r["replan"] for r in rows) / n
        retry_wins = sum(1 for r in rows if r["retry"] > r["continue"])
        replan_wins = sum(1 for r in rows if r["replan"] > r["continue"])
        print(f"{outcome:8} {n:>5} {continue_mean:>14.4f} {retry_mean:>12.4f} {replan_mean:>13.4f} "
              f"{retry_wins/n*100:>11.1f}% {replan_wins/n*100:>12.1f}%")

    all_rows = results["won"] + results["lost"]
    if not all_rows:
        print("\nno usable comparisons collected.")
        return

    n = len(all_rows)
    continue_scores = [r["continue"] for r in all_rows]
    retry_scores = [r["retry"] for r in all_rows]
    replan_scores = [r["replan"] for r in all_rows]
    continue_mean, retry_mean, replan_mean = (sum(s) / n for s in (continue_scores, retry_scores, replan_scores))
    print(f"\nOVERALL (n={n}): continue_mean={continue_mean:.4f}, retry_mean={retry_mean:.4f}, "
          f"replan_mean={replan_mean:.4f}")

    # separability check (finding 2 from the coordinator_v2 postmortem review): pooled std across
    # all three actions' scores, vs. the spread between their means. if the between-action mean
    # spread is small relative to the pooled std, the reward is NOT separable at this n -- PPO has
    # nothing reliable to learn from regardless of policy-side fixes (zero-init, entropy, etc).
    pooled = continue_scores + retry_scores + replan_scores
    pooled_std = statistics.pstdev(pooled) if len(pooled) > 1 else 0.0
    mean_spread = max(continue_mean, retry_mean, replan_mean) - min(continue_mean, retry_mean, replan_mean)
    print(f"pooled_std={pooled_std:.4f}, mean_spread_across_actions={mean_spread:.4f}, "
          f"spread/std_ratio={mean_spread / pooled_std if pooled_std > 1e-9 else float('nan'):.3f}")
    print(
        "\nif spread/std_ratio is small (well under ~0.3-0.5), the three actions' rewards are NOT "
        "meaningfully separable at this sample scale -- no coordinator-side fix (zero-init, "
        "entropy coefficient, KL, anything) can make PPO learn real discrimination from noise "
        "this close to indistinguishable. the fix would be more episodes/iteration (train_ppo.py "
        "currently uses 20) or a different reward design, not another policy/hyperparameter change."
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
