"""
collect BRANCHING rollouts for verifier_v6: at K decision points per episode, sample 2 DIFFERENT
worker actions from the SAME state, let each branch continue independently to episode end, and
record each branch's own REAL Monte-Carlo return-to-go. produces genuine (state, action_A, G_A)
vs (state, action_B, G_B) pairs where G_A != G_B reflects a real downstream outcome difference --
the signal missing from the single-linear-trajectory data verifier_v1-v5 were trained on.

root cause this targets (see .info/CLAUDE.md's build 13 section, "verifier_v6" note): verifier_v5's
advantage label is `q_value_t - q_value_{t-1}`, which does NOT depend on the action taken at turn
t (q_value_{t-1} is the PREVIOUS turn's return, unaffected by what's chosen next) -- confirmed via
direct algebraic derivation as the reason only ~11-14% of coordinator_v5's counterfactual reward
comparisons came out nonzero. this script does not fix the labeling formula by itself; it supplies
the missing PAIRED data so verifier/train.py can add a pairwise ranking loss term (see that file's
--branch-pairs flag) that directly penalizes the model for scoring two different-outcome actions
at the same state too similarly.

mechanism: run a normal episode first to get a real trajectory of length T. pick K decision points
(evenly spaced, skipping the first few and last few turns so branches have real tails to diverge
over). at each point, REPLAY the identical action prefix from a fresh env reset (TextWorldExpress's
reset(seed=...) is deterministic -- confirmed in envs/textworldexpress_env.py), then diverge: call
worker.act() twice at the branch state -- once normally (branch A), once with branch A's action
masked out of the admissible list (branch B, forcing a genuinely different choice, the same
masking mechanism already validated for "retry" in rollout/runner.py). let each branch play out
independently to episode end (or step cap), compute each branch's own real MC return-to-go from
that point forward.

cost: TextWorldExpress env stepping is cheap; the real cost is Worker LLM calls, roughly
2 x (average branch tail length) extra calls per branch point. local Qwen2.5-3B only, per this
project's standing cost-discipline rule -- no exceptions.

usage:
    python -m scripts.collect_branches --n-episodes 150 --branches-per-episode 3 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out data/rollouts/v6_branches \\
        --split train
"""

import argparse
import json
import os
import random

from dotenv import load_dotenv
load_dotenv()

from agents import thinker, worker
from envs.textworldexpress_env import TextWorldExpressEnvWrapper, TRAINING_GAMES

GAMMA = 0.99


def _mc_return(rewards: list[float], gamma: float = GAMMA) -> float:
    """discounted sum of a reward sequence from its own start -- G_0 for a branch tail."""
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
    return running


def _run_prefix(env, game: str, seed: int, prefix_actions: list[str]) -> tuple[str, dict, list[str]]:
    """deterministically replay env to the state right after prefix_actions. returns
    (obs, info, action_history) at that point -- the shared branch point both A and B diverge
    from."""
    obs, info = env.reset_to(game, seed)
    action_history: list[str] = []
    for a in prefix_actions:
        obs, _, done, info = env.step(a)
        action_history.append(a)
        if done:
            break  # prefix shouldn't reach done in practice (branch points are picked mid-episode)
    return obs, info, action_history


def _play_branch(
    env, task_goal: str, plan: str, obs: str, info: dict, action_history: list[str],
    first_action: str, worker_model: str, max_steps: int,
) -> tuple[list[str], list[float]]:
    """execute first_action, then let the worker continue normally to episode end (or step cap).
    returns (actions_taken, rewards) for this branch's OWN tail, starting from first_action."""
    actions: list[str] = []
    rewards: list[float] = []

    next_obs, reward, done, next_info = env.step(first_action)
    actions.append(first_action)
    rewards.append(reward)
    action_history = action_history + [first_action]
    steps = len(action_history)

    obs, info = next_obs, next_info
    while not done and steps < max_steps:
        admissible = env.admissible_commands(info)
        chosen, _ = worker.act(task_goal, plan, obs, admissible, action_history, model=worker_model)
        obs, reward, done, info = env.step(chosen)
        actions.append(chosen)
        rewards.append(reward)
        action_history = action_history + [chosen]
        steps += 1

    return actions, rewards


def collect(
    n_episodes: int, branches_per_episode: int, worker_model: str, out_dir: str, split: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pair_id = 0
    seed_counter = 0

    for ep_i in range(n_episodes):
        game = TRAINING_GAMES[ep_i % len(TRAINING_GAMES)]
        seed_counter += 1
        seed = seed_counter

        # 1. run a normal reference episode to know its real length and get a real plan
        env = TextWorldExpressEnvWrapper(split=split, games=(game,))
        obs, info = env.reset_to(game, seed)
        task_goal = env.task_goal(obs, info)
        plan, _ = thinker.plan(task_goal, obs, model=worker_model)

        action_history: list[str] = []
        max_steps = getattr(env, "step_limit", 50)
        done = False
        cur_obs = obs
        while not done and len(action_history) < max_steps:
            admissible = env.admissible_commands(info)
            chosen, _ = worker.act(task_goal, plan, cur_obs, admissible, action_history, model=worker_model)
            cur_obs, _, done, info = env.step(chosen)
            action_history.append(chosen)
        episode_len = len(action_history)
        env.close()

        if episode_len < 6:
            print(f"  [{ep_i+1}/{n_episodes}] {game} episode too short ({episode_len} steps), skipping branches")
            continue

        # 2. pick K branch points, evenly spaced, skipping the first/last 2 turns (need a real
        # prefix to replay and a real tail to diverge over)
        lo, hi = 2, episode_len - 2
        if hi <= lo:
            print(f"  [{ep_i+1}/{n_episodes}] {game} too short for branch margins, skipping")
            continue
        n_points = min(branches_per_episode, hi - lo)
        branch_points = sorted(random.sample(range(lo, hi), n_points))

        for bp in branch_points:
            prefix = action_history[:bp]

            # branch env: fresh instance per branch point, replay identical prefix deterministically
            env_a = TextWorldExpressEnvWrapper(split=split, games=(game,))
            branch_obs, branch_info, branch_history = _run_prefix(env_a, game, seed, prefix)
            admissible = env_a.admissible_commands(branch_info)
            if len(admissible) < 2:
                env_a.close()
                continue

            # branch A: worker's normal choice at this state
            action_a, _ = worker.act(task_goal, plan, branch_obs, admissible, branch_history, model=worker_model)
            tail_actions_a, tail_rewards_a = _play_branch(
                env_a, task_goal, plan, branch_obs, branch_info, branch_history,
                action_a, worker_model, max_steps,
            )
            env_a.close()

            # branch B: SAME state, action_a masked out, forces a genuinely different choice
            # (same masking mechanism already validated for "retry" in rollout/runner.py)
            env_b = TextWorldExpressEnvWrapper(split=split, games=(game,))
            branch_obs_b, branch_info_b, branch_history_b = _run_prefix(env_b, game, seed, prefix)
            worker_choices_b = [c for c in admissible if c != action_a]
            if not worker_choices_b:
                env_b.close()
                continue
            action_b, _ = worker.act(
                task_goal, plan, branch_obs_b, worker_choices_b, branch_history_b, model=worker_model,
            )
            tail_actions_b, tail_rewards_b = _play_branch(
                env_b, task_goal, plan, branch_obs_b, branch_info_b, branch_history_b,
                action_b, worker_model, max_steps,
            )
            env_b.close()

            g_a = _mc_return(tail_rewards_a)
            g_b = _mc_return(tail_rewards_b)

            record = {
                "pair_id": pair_id,
                "game": game,
                "task_goal": task_goal,
                "plan": plan,
                "obs_before": branch_obs,
                "action_history": branch_history,
                "branch_a": {"action": action_a, "q_value": round(g_a, 6)},
                "branch_b": {"action": action_b, "q_value": round(g_b, 6)},
            }
            with open(os.path.join(out_dir, f"pair_{pair_id:05d}.json"), "w") as f:
                json.dump(record, f, indent=2)
            pair_id += 1

            print(f"  [{ep_i+1}/{n_episodes}] {game} bp={bp}: "
                  f"A={action_a!r} G_A={g_a:.4f} | B={action_b!r} G_B={g_b:.4f} | "
                  f"diff={abs(g_a-g_b):.4f}", flush=True)

    print(f"\ncollected {pair_id} branch pairs -> {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=150)
    parser.add_argument("--branches-per-episode", type=int, default=3)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", type=str, default="data/rollouts/v6_branches")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    collect(args.n_episodes, args.branches_per_episode, args.worker_model, args.out, args.split)
