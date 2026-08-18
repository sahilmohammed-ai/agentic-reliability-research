"""
Best-of-N verifier-guided evaluation (DESIGN_SPEC.md Phase 4) -- the main evidence for whether
either trained verifier's signal actually improves real-time decisions, independent of any
learned policy or RL training.

mechanism: at each worker turn, sample N candidate actions from the SAME frozen worker model
(temperature > 0, genuinely different candidates -- greedy decoding would return the identical
action N times, see agents/llm.py's temperature parameter added for exactly this), score each
candidate with a verifier, execute the argmax-scoring one. compares against N=1 greedy
(rollout/runner.py's run_episode, completely unmodified zero-coordination baseline) on the same
held-out eval_ood set used throughout the verifier evaluation history.

what this tests and does NOT test: this only needs the verifier to rank a handful of DIFFERENT
candidates against each other at the SAME turn -- the coarse, aggregate-level discrimination both
verifier_mc (AUC 0.56-0.84, MC-return-labeled, balanced training data) and verifier_dpo
(92.99% episode-level held-out accuracy, preference-trained) have already demonstrated. it does
NOT require fine-grained same-state action discrimination the way a policy-gradient reward would
-- Best-of-N just needs "of these N options, which does the verifier prefer," which is exactly
the kind of discrimination the AUC/accuracy numbers already certify. real result: a win-rate lift
on coin (0%->65% with verifier_dpo, 0%->35% with verifier_mc), the first result in this project
to move that game off zero, direct evidence the verifier's signal is applicable, not just
statistically separable offline.

supports scoring candidates with any of FOUR --scorer options:
  - mc: verifier.infer.Verifier (checkpoints/verifier_mc), uses .advantage on each candidate turn.
  - dpo: verifier_dpo's PreferenceScorer, uses the mean per-token score of the trajectory-so-far
    WITH the candidate action appended (a growing-prefix episode_score call), since verifier_dpo
    was trained on whole-episode text, not isolated single turns -- scoring a bare single turn in
    verifier_mc's per-turn format would be off-distribution for it (the same train/inference
    mismatch class of bug already found and fixed once in verifier_dpo/infer.py).
  - llm: verifier.frozen_llm.score_candidate(), a genuinely NEW variant of build 08/09's frozen
    LLM judge (added 2026-08-04) -- the original score_turn() judges a turn using obs_after (the
    RESULT of an action already taken), which Best-of-N structurally cannot provide (candidates
    must be ranked BEFORE one is executed). score_candidate() is a separate function/prompt that
    predicts a candidate's quality from intent alone (no obs_after), a different task from the one
    build 08/09 validated -- its own numbers here are this task's first real evaluation, not
    inherited from that earlier validation. no local model to load; each candidate costs one real
    LLM inference call (slower than mc/dpo's local forward passes).
  - random: the NO-VERIFIER ablation (added 2026-08-06). candidates are still sampled from the
    worker at temperature>0 (same diversity mechanism as every other scorer), but each candidate
    gets a random score, so the argmax pick is effectively arbitrary among them. isolates how much
    of Best-of-N's win-rate lift comes from the VERIFIER'S JUDGMENT specifically, vs. just the act
    of sampling N candidates and picking SOME one of them (sampling alone might occasionally
    surface a better action than a single greedy call would, even before any real scoring is
    applied -- this scorer measures exactly that effect in isolation). zero model/checkpoint
    needed, the cheapest run in this script.

there is no per-candidate "accuracy" metric here -- no ground-truth optimal action exists to score
predictions against. the metric is WIN RATE: does letting the verifier pick among N sampled
candidates each turn win more often, in aggregate over many episodes, than a plain single greedy
choice (see the baseline vs. best_of_n win-rate comparison this script prints/saves).

usage:
    python -m scripts.best_of_n_eval --scorer mc --n 5 --episodes-per-game 20 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_mc.json
    python -m scripts.best_of_n_eval --scorer dpo --n 5 --episodes-per-game 20 \\
        --dpo-checkpoint verifier_dpo/checkpoints/finetune_v2/model.pt \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_dpo.json
    python -m scripts.best_of_n_eval --scorer llm --n 5 --episodes-per-game 20 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_llm.json
    python -m scripts.best_of_n_eval --scorer random --n 5 --episodes-per-game 20 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct --out reports/best_of_n_random.json
"""

import argparse
import json
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from agents import single_agent, thinker, worker
from envs.textworldexpress_env import TextWorldExpressEnvWrapper, TRAINING_GAMES
from rollout.schemas import Trajectory, Turn

TEMPERATURE = 0.8  # standard, moderate sampling temperature for real candidate diversity
                    # without degenerating into near-random/incoherent actions


def _score_candidates_mc(verifier_obj, task_goal, plan, obs, candidates):
    """returns per-candidate advantage scores via verifier.infer.Verifier.score_batch()."""
    items = [(task_goal, plan, obs, action) for action in candidates]
    results = verifier_obj.score_batch(items)  # list of (q_value, advantage)
    return [advantage for _, advantage in results]


def _score_candidates_llm(judge_model, task_goal, action_history, obs, candidates):
    """returns per-candidate scores via verifier.frozen_llm.score_candidate() -- a real LLM
    reasoning call per candidate, no local model needed but much slower than mc/dpo's local
    forward passes (one live inference call per candidate, not a batched tensor op)."""
    from verifier.frozen_llm import score_candidate
    scores = []
    for action in candidates:
        score, _usage = score_candidate(task_goal, action_history, obs, action, model=judge_model)
        scores.append(score)
    return scores


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
        scores = score_fn(task_goal, ep_plan, current_obs, candidates, traj_so_far, action_history)
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


def run_best_of_n_single_agent_episode(
    env, n: int, worker_model: str, score_fn, task_id: str | None = None,
) -> Trajectory:
    """single-agent counterpart to run_best_of_n_episode: mirrors rollout/runner.py's
    run_single_agent_episode() (no thinker, no plan, one model call per step via
    agents/single_agent.py) instead of the thinker+worker loop. sample N candidates from
    single_agent.act() at temperature>0, score with the verifier, execute the argmax -- same
    mechanism as the thinker+worker version, just applied to the plan-free baseline requested for
    a same-settings comparison against build 01's single-agent numbers.

    plan is passed to score_fn as "" throughout (never None), matching how
    rollout/runner.py's run_single_agent_episode() stores plan="" for its own trajectories --
    verifier.infer.Verifier.score() and verifier_dpo's build_episode_text() both accept an empty
    plan string as a real, valid input, not a special case to branch on."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    max_steps = getattr(env, "step_limit", 50)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    done = False
    current_obs = obs
    traj_so_far = {"task_goal": task_goal, "plan": "", "turns": []}

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        seen, candidates = set(), []
        for _ in range(n):
            cand, _cand_usage = single_agent.act(
                task_goal, current_obs, admissible, action_history,
                model=worker_model, env_hint=env_hint, temperature=TEMPERATURE,
            )
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)
        if not candidates:
            candidates = [admissible[0]]

        traj_so_far["_pending_obs"] = current_obs
        scores = score_fn(task_goal, "", current_obs, candidates, traj_so_far, action_history)
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        chosen = candidates[best_idx]

        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turns.append(Turn(
            step=env_step, role="worker", obs_before=current_obs, action=chosen,
            obs_after=next_obs, env_reward=reward, done=done,
            metadata={
                "admissible_commands": admissible,
                "best_of_n_candidates": candidates,
                "best_of_n_scores": scores,
                "best_of_n_chosen_idx": best_idx,
            },
        ))
        traj_so_far["turns"].append({
            "role": "worker", "step": env_step, "obs_before": current_obs, "action": chosen,
        })
        current_obs = next_obs

    return Trajectory(
        task_id=task_id, task_goal=task_goal, plan="", turns=turns,
        won=env.won(info), total_steps=env_step,
    )


def run_baseline_episode(env, worker_model: str, task_id: str | None = None) -> Trajectory:
    """N=1 greedy baseline (thinker+worker loop) -- delegates to rollout/runner.py's run_episode()
    directly rather than reimplementing it, so this is provably the exact same code path builds
    03/10-12 were evaluated against, not a re-derived approximation of it."""
    from rollout.runner import run_episode
    return run_episode(env, task_id=task_id, model=worker_model)


def run_baseline_single_agent_episode(env, worker_model: str, task_id: str | None = None) -> Trajectory:
    """N=1 greedy baseline (single-agent, no plan) -- delegates to rollout/runner.py's
    run_single_agent_episode() directly, the exact code path build 01 was evaluated against."""
    from rollout.runner import run_single_agent_episode
    return run_single_agent_episode(env, task_id=task_id, model=worker_model)


def evaluate(
    scorer: str, n: int, episodes_per_game: int, worker_model: str,
    checkpoint_dir: str, dpo_checkpoint: str, split: str, out_path: str,
    games: tuple[str, ...] | None = None, agent_type: str = "loop",
) -> None:
    """NOTE on methodology: baseline and best-of-n episodes are run on INDEPENDENT env resets
    (TextWorldExpressEnvWrapper.reset() draws a new random seed each call, confirmed in
    envs/textworldexpress_env.py) -- the i-th baseline episode and the i-th best-of-n episode are
    NOT the same underlying task instance, matching how builds 03/10-12 were evaluated (each run
    draws fresh, no fixed-seed pairing). only the AGGREGATE win rate over episodes_per_game is a
    meaningful comparison; individual per-episode print lines are not a controlled pairwise A/B.

    agent_type="loop" (default): thinker+worker agentic loop, the ORIGINAL Best-of-N condition.
    agent_type="single": the plan-free single-agent baseline (build 01), requested as a same-
    settings comparison alongside the loop -- does Best-of-N help even without a thinker's plan?"""
    games = games or TRAINING_GAMES
    if agent_type == "loop":
        run_bon_episode, run_base_episode = run_best_of_n_episode, run_baseline_episode
    elif agent_type == "single":
        run_bon_episode, run_base_episode = run_best_of_n_single_agent_episode, run_baseline_single_agent_episode
    else:
        raise ValueError(f"unknown agent_type: {agent_type}")

    if scorer == "mc":
        from verifier.infer import Verifier
        verifier_obj = Verifier(checkpoint_dir)

        def score_fn(task_goal, plan, obs, candidates, _traj_so_far, _action_history):
            return _score_candidates_mc(verifier_obj, task_goal, plan, obs, candidates)

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

        def score_fn(task_goal, plan, obs, candidates, traj_so_far, _action_history):
            return _score_candidates_dpo(model, tokenizer, device, traj_so_far, candidates)

    elif scorer == "llm":
        # no local model to load -- score_candidate makes a real inference call per candidate
        # (via agents/llm.py's complete_with_usage), same worker_model used for the worker itself
        # unless a different judge model is wanted; reusing worker_model keeps this comparable to
        # build 08/09's frozen-LLM judge, which also used the project's standard local model.
        def score_fn(task_goal, plan, obs, candidates, _traj_so_far, action_history):
            return _score_candidates_llm(worker_model, task_goal, action_history, obs, candidates)

    elif scorer == "random":
        # the no-verifier ablation: candidates are still sampled at temperature>0 (same diversity
        # mechanism as every other scorer), but the argmax pick is effectively arbitrary since
        # every candidate gets a random score. isolates how much of Best-of-N's win-rate lift comes
        # from the VERIFIER'S JUDGMENT specifically, vs. just the act of sampling N candidates and
        # picking SOME one of them (e.g. sampling might itself surface a better action than a
        # single greedy call would, even before any scoring is applied). zero model/checkpoint
        # needed -- the cheapest possible run in this script.
        import random as _random

        def score_fn(task_goal, plan, obs, candidates, _traj_so_far, _action_history):
            return [_random.random() for _ in candidates]

    else:
        raise ValueError(f"unknown scorer: {scorer}")

    results = {"scorer": scorer, "n": n, "agent_type": agent_type, "games": {}}

    # save Best-of-N episode traces (candidates + scores at every turn, plus the plain baseline
    # trajectory for comparison) -- added after the first real run's results.json turned out to
    # contain only aggregate win rates, with no way to inspect WHY a game like mapreader stayed at
    # 0% under both conditions (worker capability gap vs. some best-of-n-specific issue) without
    # rerunning. traces_dir defaults alongside out_path so this doesn't need a new CLI flag.
    # agent_type suffix keeps loop and single-agent traces from overwriting each other.
    traces_dir = os.path.join(os.path.dirname(out_path) or ".", f"best_of_n_traces_{scorer}_{agent_type}")
    os.makedirs(traces_dir, exist_ok=True)

    for game in games:
        baseline_won, bon_won = 0, 0
        for i in range(episodes_per_game):
            env = TextWorldExpressEnvWrapper(split=split, games=(game,))
            baseline_traj = run_base_episode(env, worker_model, task_id=f"{game}_baseline_{i}")
            baseline_won += int(baseline_traj.won)
            env.close()

            env = TextWorldExpressEnvWrapper(split=split, games=(game,))
            bon_traj = run_bon_episode(
                env, n, worker_model, score_fn, task_id=f"{game}_bon_{i}",
            )
            bon_won += int(bon_traj.won)
            env.close()

            with open(os.path.join(traces_dir, f"{game}_{i:03d}.json"), "w") as f:
                json.dump({
                    "baseline": baseline_traj.to_dict(),
                    "best_of_n": bon_traj.to_dict(),
                }, f, indent=2)

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
    parser.add_argument("--scorer", choices=["mc", "dpo", "llm", "random"], required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--episodes-per-game", type=int, default=20)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/verifier_mc")
    parser.add_argument("--dpo-checkpoint", type=str,
                         default="verifier_dpo/checkpoints/finetune_v2/model.pt")
    parser.add_argument("--split", type=str, default="eval_ood")
    parser.add_argument("--out", type=str, default="reports/best_of_n_results.json")
    parser.add_argument("--games", type=str, default=None,
                         help="comma-separated subset of TRAINING_GAMES (default: all 4) -- "
                              "e.g. --games mapreader to re-check one game's traces cheaply "
                              "without rerunning the full sweep")
    parser.add_argument("--agent-type", choices=["loop", "single"], default="loop",
                         help="loop (default): thinker+worker agentic loop, the original "
                              "Best-of-N condition. single: the plan-free single-agent baseline "
                              "(build 01) -- same Best-of-N mechanism, no thinker/plan involved.")
    args = parser.parse_args()
    games = tuple(args.games.split(",")) if args.games else None
    evaluate(
        args.scorer, args.n, args.episodes_per_game, args.worker_model,
        args.checkpoint_dir, args.dpo_checkpoint, args.split, args.out, games=games,
        agent_type=args.agent_type,
    )
