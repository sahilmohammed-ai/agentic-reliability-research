"""
evaluate a trained coordinator checkpoint: run it GREEDILY (deterministic argmax, no sampling --
this is deployment/eval behavior, not the stochastic rollouts train_ppo.py collects) on fresh
eval_ood episodes, and compare win rate against the zero-coordination baseline (rollout.runner.
run_episode) on the SAME game mix and episode count, so the comparison isolates what the learned
coordinator itself contributes.

both conditions run on --split eval_ood, never touched by verifier_v5's training (build 10-12) or
by this coordinator's own PPO rollouts (train_ppo.py used --split train, see its collect_batch).

usage:
    python -m coordinator.evaluate --coordinator-checkpoint checkpoints/coordinator_v1 \\
        --verifier-checkpoint checkpoints/verifier_v5 --n-per-game 30 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out data/labeled/coordinator_v1_eval.json
"""

import argparse
import json
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from coordinator.infer import CoordinatorPolicy
from envs.textworldexpress_env import TextWorldExpressEnvWrapper, TRAINING_GAMES
from rollout.runner import run_episode, run_learned_coordinated_episode
from verifier.infer import Verifier


def run_condition(
    label: str, games: tuple[str, ...], n_per_game: int, worker_model: str, split: str,
    episode_fn,
) -> dict:
    """run n_per_game episodes per game with episode_fn(env, task_id=..., model=...) -> Trajectory.
    returns {game: {"won": int, "total": int, "win_rate": float, "avg_steps": float}}."""
    results = defaultdict(lambda: {"won": 0, "total": 0, "steps": []})
    for game in games:
        env = TextWorldExpressEnvWrapper(split=split, games=(game,))
        for i in range(n_per_game):
            traj = episode_fn(env, task_id=f"{label}_{game}_{i}", model=worker_model)
            results[game]["won"] += int(traj.won)
            results[game]["total"] += 1
            results[game]["steps"].append(traj.total_steps)
            print(f"  [{label}] {game} {i+1}/{n_per_game}: won={traj.won} steps={traj.total_steps}", flush=True)
        env.close()

    summary = {}
    for game, r in results.items():
        summary[game] = {
            "won": r["won"],
            "total": r["total"],
            "win_rate": r["won"] / r["total"],
            "avg_steps": sum(r["steps"]) / len(r["steps"]),
        }
    return summary


def main(
    coordinator_checkpoint: str, verifier_checkpoint: str, n_per_game: int,
    worker_model: str, out_path: str, split: str = "eval_ood",
) -> None:
    coordinator = CoordinatorPolicy(coordinator_checkpoint)
    verifier = Verifier(verifier_checkpoint, bound_q_value=False)

    def learned_episode_fn(env, task_id, model):
        return run_learned_coordinated_episode(
            env, coordinator=coordinator, verifier=verifier, task_id=task_id, model=model, greedy=True,
        )

    print(f"\n=== condition: learned coordinator ({coordinator_checkpoint}) ===", flush=True)
    learned_results = run_condition(
        "learned", TRAINING_GAMES, n_per_game, worker_model, split, learned_episode_fn,
    )

    print(f"\n=== condition: zero-coordination baseline ===", flush=True)
    baseline_results = run_condition(
        "baseline", TRAINING_GAMES, n_per_game, worker_model, split, run_episode,
    )

    print("\n=== comparison ===", flush=True)
    print(f"{'game':14} {'baseline win%':>14} {'learned win%':>14} {'delta':>8}", flush=True)
    overall_baseline_won = overall_baseline_total = 0
    overall_learned_won = overall_learned_total = 0
    for game in TRAINING_GAMES:
        b = baseline_results[game]
        l = learned_results[game]
        delta = l["win_rate"] - b["win_rate"]
        print(f"{game:14} {b['win_rate']*100:>13.1f}% {l['win_rate']*100:>13.1f}% {delta*100:>+7.1f}%", flush=True)
        overall_baseline_won += b["won"]
        overall_baseline_total += b["total"]
        overall_learned_won += l["won"]
        overall_learned_total += l["total"]

    overall_b = overall_baseline_won / overall_baseline_total
    overall_l = overall_learned_won / overall_learned_total
    print(f"{'OVERALL':14} {overall_b*100:>13.1f}% {overall_l*100:>13.1f}% {(overall_l-overall_b)*100:>+7.1f}%", flush=True)

    with open(out_path, "w") as f:
        json.dump({"baseline": baseline_results, "learned": learned_results}, f, indent=2)
    print(f"\nsaved detailed results to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-checkpoint", type=str, required=True)
    parser.add_argument("--verifier-checkpoint", type=str, default="checkpoints/verifier_v5")
    parser.add_argument("--n-per-game", type=int, default=30)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval_ood")
    args = parser.parse_args()
    main(
        args.coordinator_checkpoint, args.verifier_checkpoint, args.n_per_game,
        args.worker_model, args.out, args.split,
    )
