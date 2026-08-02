"""
Best-of-N verifier-guided evaluation (DESIGN_SPEC.md Phase 4 / .info/CLAUDE.MD's staged plan,
step 2), the cheap validation step before spending coordinator/PPO time on either verifier again.

mechanism: at each worker turn, sample N candidate actions from the SAME frozen worker model
(temperature > 0, genuinely different candidates -- greedy decoding would return the identical
action N times, see agents/llm.py's temperature parameter added for exactly this), score each
candidate with a verifier, execute the argmax-scoring one. compares against N=1 greedy
(rollout/runner.py's run_episode, completely unmodified zero-coordination baseline) on the SAME
eval_ood set builds 10-12 used.

what this tests and does NOT test: this only needs the verifier to rank a handful of DIFFERENT
candidates against each other at the SAME turn -- the coarse, aggregate-level discrimination both
verifier_v5 (AUC 0.56-0.84, build 12) and verifier_dpo (92.99% episode-level held-out accuracy,
solidification sweep) have already demonstrated. it does NOT require the same-state RL-reward
resolution build 13's coordinator diagnosis showed is unavailable in this environment -- Best-of-N
just needs "of these N options, which does the verifier prefer," not a calibrated numeric
advantage suitable for a policy-gradient reward. a real win-rate lift here is the evidence needed
before justifying another coordinator/PPO cycle; no lift is a fast, cheap negative result that
saves a multi-day GPU run on a premise this script can falsify in hours.

supports scoring candidates with EITHER verifier via --scorer:
  - v5: verifier.infer.Verifier (checkpoints/verifier_v5), uses .advantage on each candidate turn.
  - dpo: verifier_dpo's PreferenceScorer, uses the mean per-token score of the trajectory-so-far
    WITH the candidate action appended (a growing-prefix episode_score call), since verifier_dpo
    was trained on whole-episode text, not isolated single turns -- scoring a bare single turn in
    verifier_v5's per-turn format would be off-distribution for it (the same train/inference
    mismatch class of bug already found and fixed once in verifier_dpo/infer.py).

usage:
    python -m scripts.best_of_n_eval --scorer v5 --n 5 --episodes-per-game 20 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_v5.json
    python -m scripts.best_of_n_eval --scorer dpo --n 5 --episodes-per-game 20 \\
        --dpo-checkpoint verifier_dpo/checkpoints/finetune_v2/model.pt \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_dpo.json
"""

import argparse
import json
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from agents import thinker, worker
from envs.textworldexpress_env import TextWorldExpressEnvWrapper, TRAINING_GAMES
from rollout.schemas import Trajectory, Turn

TEMPERATURE = 0.8  # standard, moderate sampling temperature for real candidate diversity
                    # without degenerating into near-random/incoherent actions


def _score_candidates_v5(verifier_obj, task_goal, plan, obs, candidates):
    """returns per-candidate advantage scores via verifier.infer.Verifier.score_batch()."""
    items = [(task_goal, plan, obs, action) for action in candidates]
    results = verifier_obj.score_batch(items)  # list of (q_value, advantage)
    return [advantage for _, advantage in results]


def _score_candidates_dpo(model, tokenizer, device, traj_so_far: dict, candidates):
    """scores each candidate by appending it as one more turn to the trajectory-so-far and
    computing episode_score() on the resulting growing-prefix text -- verifier_dpo was trained on
    whole-episode text (see verifier_dpo/dataset.py's build_episode_text), so scoring it here in
    that same format (a real prefix of an episode, not an isolated single-turn string) keeps this
    in-distribution rather than repeating the train/inference mismatch already found once in
    verifier_dpo/infer.py before it was fixed."""
    import torch
    from verifier_dpo.dataset import build_episode_text

    scores = []
    next_step = len(traj_so_far["turns"]) + 1
    for action in candidates:
        candidate_traj = dict(traj_so_far)
        candidate_traj["turns"] = traj_so_far["turns"] + [{
            "role": "worker", "step": next_step,
            "obs_before": traj_so_far["_pending_obs"], "action": action,
        }]
        text, _ = build_episode_text(candidate_traj)
        enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            score = model.episode_score(enc["input_ids"], enc["attention_mask"])
        scores.append(score.item())
    return scores


def run_best_of_n_episode(
    env, n: int, worker_model: str, score_fn, task_id: str | None = None,
) -> Trajectory:
    """mirrors rollout/runner.py's run_episode() exactly (same plan-once, act-to-done loop, same
    Turn/Trajectory shape), except at each worker turn it samples N candidates and executes the
    verifier-argmax instead of a single greedy call. any win-rate delta vs. run_episode isolates
    the effect of Best-of-N selection, not a different episode structure."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    max_steps = getattr(env, "step_limit", 50)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=worker_model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage
    traj_so_far = {"task_goal": task_goal, "plan": ep_plan, "turns": []}

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        # sample N candidates at temperature > 0 -- deduplicate while preserving order, since a
        # short admissible list can produce the same action more than once by chance, which would
        # waste a verifier call without adding real diversity.
        seen, candidates, usages = set(), [], []
        for _ in range(n):
            cand, cand_usage = worker.act(
                task_goal, ep_plan, current_obs, admissible, action_history,
                model=worker_model, temperature=TEMPERATURE,
            )
            usages.append(cand_usage)
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)
        if not candidates:
            candidates = [admissible[0]]

        traj_so_far["_pending_obs"] = current_obs
        scores = score_fn(task_goal, ep_plan, current_obs, candidates, traj_so_far)
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        chosen = candidates[best_idx]

        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turn_metadata = {
            "admissible_commands": admissible,
            "usage": usages[0],  # first sample's usage kept for cost tracking, matches run_episode's shape
            "best_of_n_candidates": candidates,
            "best_of_n_scores": scores,
            "best_of_n_chosen_idx": best_idx,
        }
        if pending_plan_usage is not None:
            turn_metadata["plan_usage"] = pending_plan_usage
            pending_plan_usage = None

        turns.append(Turn(
            step=env_step, role="worker", obs_before=current_obs, action=chosen,
            obs_after=next_obs, env_reward=reward, done=done, metadata=turn_metadata,
        ))
        traj_so_far["turns"].append({
            "role": "worker", "step": env_step, "obs_before": current_obs, "action": chosen,
        })
        current_obs = next_obs

    return Trajectory(
        task_id=task_id, task_goal=task_goal, plan=ep_plan, turns=turns,
        won=env.won(info), total_steps=env_step,
    )


def run_baseline_episode(env, worker_model: str, task_id: str | None = None) -> Trajectory:
    """N=1 greedy baseline -- delegates to rollout/runner.py's run_episode() directly rather than
    reimplementing it, so this is provably the exact same code path builds 03/10-12 were evaluated
    against, not a re-derived approximation of it."""
    from rollout.runner import run_episode
    return run_episode(env, task_id=task_id, model=worker_model)


def evaluate(
    scorer: str, n: int, episodes_per_game: int, worker_model: str,
    checkpoint_dir: str, dpo_checkpoint: str, split: str, out_path: str,
) -> None:
    """NOTE on methodology: baseline and best-of-n episodes are run on INDEPENDENT env resets
    (TextWorldExpressEnvWrapper.reset() draws a new random seed each call, confirmed in
    envs/textworldexpress_env.py) -- the i-th baseline episode and the i-th best-of-n episode are
    NOT the same underlying task instance, matching how builds 03/10-12 were evaluated (each run
    draws fresh, no fixed-seed pairing). only the AGGREGATE win rate over episodes_per_game is a
    meaningful comparison; individual per-episode print lines are not a controlled pairwise A/B."""
    games = TRAINING_GAMES

    if scorer == "v5":
        from verifier.infer import Verifier
        verifier_obj = Verifier(checkpoint_dir)

        def score_fn(task_goal, plan, obs, candidates, _traj_so_far):
            return _score_candidates_v5(verifier_obj, task_goal, plan, obs, candidates)

    elif scorer == "dpo":
        import torch
        from transformers import AutoTokenizer
        from verifier_dpo.model import BASE_MODEL, PreferenceScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = PreferenceScorer(freeze_backbone=True).to(device)
        model.load_state_dict(torch.load(dpo_checkpoint, map_location=device))
        model.eval()

        def score_fn(task_goal, plan, obs, candidates, traj_so_far):
            return _score_candidates_dpo(model, tokenizer, device, traj_so_far, candidates)

    else:
        raise ValueError(f"unknown scorer: {scorer}")

    results = {"scorer": scorer, "n": n, "games": {}}

    for game in games:
        baseline_won, bon_won = 0, 0
        for i in range(episodes_per_game):
            env = TextWorldExpressEnvWrapper(split=split, games=(game,))
            baseline_traj = run_baseline_episode(env, worker_model, task_id=f"{game}_baseline_{i}")
            baseline_won += int(baseline_traj.won)
            env.close()

            env = TextWorldExpressEnvWrapper(split=split, games=(game,))
            bon_traj = run_best_of_n_episode(
                env, n, worker_model, score_fn, task_id=f"{game}_bon_{i}",
            )
            bon_won += int(bon_traj.won)
            env.close()

            print(f"  [{game} {i+1}/{episodes_per_game}] baseline={'WIN' if baseline_traj.won else 'lose'} "
                  f"best_of_n={'WIN' if bon_traj.won else 'lose'}", flush=True)

        baseline_rate = baseline_won / episodes_per_game
        bon_rate = bon_won / episodes_per_game
        results["games"][game] = {
            "baseline_win_rate": baseline_rate,
            "best_of_n_win_rate": bon_rate,
            "delta": bon_rate - baseline_rate,
            "n_episodes": episodes_per_game,
        }
        print(f"{game}: baseline={baseline_rate:.2%} best_of_n={bon_rate:.2%} "
              f"delta={bon_rate-baseline_rate:+.2%}\n")

    overall_baseline = sum(r["baseline_win_rate"] for r in results["games"].values()) / len(games)
    overall_bon = sum(r["best_of_n_win_rate"] for r in results["games"].values()) / len(games)
    results["overall"] = {
        "baseline_win_rate": overall_baseline,
        "best_of_n_win_rate": overall_bon,
        "delta": overall_bon - overall_baseline,
    }
    print(f"OVERALL: baseline={overall_baseline:.2%} best_of_n={overall_bon:.2%} "
          f"delta={overall_bon-overall_baseline:+.2%}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["v5", "dpo"], required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--episodes-per-game", type=int, default=20)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/verifier_v5")
    parser.add_argument("--dpo-checkpoint", type=str,
                         default="verifier_dpo/checkpoints/finetune_v2/model.pt")
    parser.add_argument("--split", type=str, default="eval_ood")
    parser.add_argument("--out", type=str, default="reports/best_of_n_results.json")
    args = parser.parse_args()
    evaluate(
        args.scorer, args.n, args.episodes_per_game, args.worker_model,
        args.checkpoint_dir, args.dpo_checkpoint, args.split, args.out,
    )
