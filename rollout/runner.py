"""
single-episode runner with a fixed (heuristic) coordinator.

four coordination modes, controlled by replan_mode:
  - "interval": thinker replans every REPLAN_INTERVAL steps on a fixed schedule (build_6).
  - "stagnation": thinker replans as soon as the worker's last STAGNATION_WINDOW actions show
    STAGNATION_MAX_UNIQUE or fewer distinct actions, i.e. it's stuck cycling (build_7). a short
    cooldown after each replan stops it from re-triggering every single step while the worker
    is still settling into the new plan.
  - "mask": same stagnation trigger as above, but on top of the replan it also masks the single
    most-repeated recent action for that one turn (build_8). replanning only suggests a new plan
    the worker can ignore; masking is a hard intervention it cannot, since the repeated action is
    removed from the admissible set for exactly that turn, then restored. drops only the single
    most-frequent action (not the whole cycle) to minimize the risk of blocking an action that
    was actually needed to win.
  - "verifier": coordinator v0. the live trained verifier (verifier/infer.py) scores every
    completed turn; an escalating low-q streak (rollout/coordinator.py) triggers one of
    continue/retry/replan/backtrack/escalate, in increasing order of cost. this is the mechanism
    validation step before any ppo training of a learned coordinator policy: the action toolkit
    must work correctly under a simple threshold rule before a policy is trained to choose
    between the same actions.

in all modes: thinker runs once at the start to generate a plan, worker acts every step.
"""

import uuid
from collections import Counter

from envs.alfworld_env import AlfWorldEnv
from agents import thinker, worker
from rollout.schemas import Turn, Trajectory
from rollout.coordinator import VerifierCoordinator, ESCALATION_MODEL

# same model used for both thinker and worker in a given run, swapped per performance test
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 50        # default episode step cap (alfworld); an env may advertise its own
                      # via a step_limit attribute (scienceworld uses 100)

REPLAN_INTERVAL = 15  # interval mode: worker steps between thinker replans

# stagnation mode: replan when the last STAGNATION_WINDOW actions contain at most
# STAGNATION_MAX_UNIQUE distinct actions (a short cycle repeating). matches the <=3-unique
# threshold already used across build_2 through build_6 to classify oscillation failures,
# so results stay comparable to that prior analysis.
STAGNATION_WINDOW = 6
STAGNATION_MAX_UNIQUE = 3
# steps to wait after a replan before stagnation can trigger again, so the thinker isn't
# re-invoked every step while the worker is still settling into the new plan (the new plan's
# first couple of actions could otherwise still look "stuck" relative to the pre-replan history)
STAGNATION_COOLDOWN = 4


def is_stagnating(action_history: list[str]) -> bool:
    """true if the worker's recent actions show a short cycle repeating."""
    if len(action_history) < STAGNATION_WINDOW:
        return False
    recent = action_history[-STAGNATION_WINDOW:]
    return len(set(recent)) <= STAGNATION_MAX_UNIQUE


def most_repeated_recent_action(action_history: list[str]) -> str | None:
    """the single most frequent action in the recent stagnation window, or None."""
    if len(action_history) < STAGNATION_WINDOW:
        return None
    recent = action_history[-STAGNATION_WINDOW:]
    return Counter(recent).most_common(1)[0][0]


def run_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
    replan_mode: str = "interval",
    verifier=None,
) -> Trajectory:
    """run one episode end-to-end with the fixed coordinator. returns a trajectory.

    verifier is a verifier.infer.Verifier instance, required when replan_mode == "verifier"
    (coordinator v0). ignored for the other modes."""
    if replan_mode == "verifier" and verifier is None:
        raise ValueError('replan_mode="verifier" requires a verifier instance (verifier.infer.Verifier)')

    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    steps_since_replan = 0  # stagnation mode's cooldown counter
    coordinator = VerifierCoordinator(verifier) if replan_mode == "verifier" else None

    # let the env advertise its own step cap if it has one (scienceworld=100), else default
    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    # each env knows where its own goal lives (alfworld embeds it in obs, scienceworld in info)
    task_goal = env.task_goal(obs, info)

    # optional per-env worker guidance (e.g. scienceworld's 'focus on' warning). envs without
    # the method contribute no hint, so alfworld is unaffected.
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    # thinker: generate plan once at episode start. the initial plan's token usage is
    # attached to the first worker turn's metadata below (there is no standalone turn for it),
    # so no per-call cost is lost.
    #
    # note: an earlier attempt (build_11) showed the thinker the admissible command list here,
    # to stop it inventing impossible actions (seen in build_9's textworldexpress run). that
    # fixed invented-command errors but, on alfworld, measurably hurt commonsense reasoning:
    # net -7 rescued/broken vs build_8, because long enumerated lists (e.g. 10+ "go to cabinet
    # N" entries) crowded out the thinker's world-knowledge guesses (e.g. "mugs are usually on
    # the countertop") in favor of picking arbitrarily from the list. reverted; see build_11
    # analysis. worth revisiting with a lighter-weight vocabulary hint (verbs only, not every
    # object instance) rather than the full raw list.
    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage  # rolled into the next worker turn's metadata

    last_worker_action: str | None = None  # for coordinator.observe(), the previous turn's action

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        # a stagnation trigger drives both the "stagnation" and "mask" modes
        stagnation_triggered = (
            replan_mode in ("stagnation", "mask")
            and steps_since_replan >= STAGNATION_COOLDOWN
            and is_stagnating(action_history)
        )

        # decide whether to replan this step, based on the active trigger mode
        if replan_mode == "interval":
            should_replan = env_step > 0 and env_step % REPLAN_INTERVAL == 0
        elif replan_mode == "verifier":
            should_replan = False  # decided below, via the coordinator's action, not here
        else:
            should_replan = stagnation_triggered

        if should_replan:
            new_plan, replan_usage = thinker.replan(task_goal, ep_plan, action_history, current_obs, model=model)
            turns.append(Turn(
                step=env_step,
                role="thinker",
                obs_before=current_obs,
                action=new_plan,
                obs_after="",
                env_reward=0.0,
                done=False,
                metadata={"type": "replan", "trigger": replan_mode, "old_plan": ep_plan, "usage": replan_usage},
            ))
            ep_plan = new_plan
            steps_since_replan = 0

        # mask mode: on a stagnation trigger, drop the single most-repeated recent action
        # from the admissible set for this one turn only, forcing a different choice. the
        # full set is restored next turn (masked_action is not carried forward).
        masked_action = None
        worker_admissible = admissible
        if replan_mode == "mask" and stagnation_triggered:
            masked_action = most_repeated_recent_action(action_history)
            if masked_action is not None:
                filtered = [c for c in admissible if c != masked_action]
                # guard: never hand the worker an empty list (only mask if something remains)
                if filtered:
                    worker_admissible = filtered
                else:
                    masked_action = None

        # coordinator v0 (replan_mode == "verifier"): score the PREVIOUS turn's outcome with the
        # live verifier, then act on the resulting low-q streak. skipped on the very first step,
        # since there is no previous worker turn yet to score.
        coordinator_action = "continue"
        coordinator_q_value = coordinator_advantage = None
        worker_model = model       # overridden to ESCALATION_MODEL only for an "escalate" turn
        worker_reconsider = False  # set True only for a "retry" turn

        if coordinator is not None and last_worker_action is not None:
            coordinator_q_value, coordinator_advantage = coordinator.observe(
                task_goal, ep_plan, current_obs, last_worker_action, action_history
            )
            coordinator_action = coordinator.decide_action()

            if coordinator_action == "retry":
                worker_reconsider = True

            elif coordinator_action == "replan":
                new_plan, replan_usage = thinker.replan(task_goal, ep_plan, action_history, current_obs, model=model)
                turns.append(Turn(
                    step=env_step, role="thinker", obs_before=current_obs, action=new_plan,
                    obs_after="", env_reward=0.0, done=False,
                    metadata={
                        "type": "replan", "trigger": "verifier", "old_plan": ep_plan, "usage": replan_usage,
                        "q_value": coordinator_q_value, "advantage": coordinator_advantage,
                    },
                ))
                ep_plan = new_plan

            elif coordinator_action == "backtrack":
                # discard stale action-history context (it is what's misleading the thinker/
                # worker) and plan fresh from the current real state, rather than revising the
                # existing plan the way replan does.
                action_history = []
                new_plan, plan_usage = thinker.plan(task_goal, current_obs, model=model)
                turns.append(Turn(
                    step=env_step, role="thinker", obs_before=current_obs, action=new_plan,
                    obs_after="", env_reward=0.0, done=False,
                    metadata={
                        "type": "backtrack", "trigger": "verifier", "old_plan": ep_plan, "usage": plan_usage,
                        "q_value": coordinator_q_value, "advantage": coordinator_advantage,
                    },
                ))
                ep_plan = new_plan

            elif coordinator_action == "escalate":
                # swap the worker model for this one turn only; reverts to the frozen worker
                # model on the next turn regardless of outcome (escalation is not sticky).
                worker_model = ESCALATION_MODEL

        # worker: select and execute one action (from the possibly-masked admissible set)
        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, worker_admissible, action_history,
            model=worker_model, env_hint=env_hint, reconsider=worker_reconsider,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        steps_since_replan += 1
        action_history.append(chosen)
        last_worker_action = chosen

        turn_metadata = {"admissible_commands": admissible, "usage": worker_usage}
        if masked_action is not None:
            turn_metadata["masked_action"] = masked_action
        if coordinator is not None:
            turn_metadata["coordinator_action"] = coordinator_action
            if coordinator_q_value is not None:
                turn_metadata["q_value"] = coordinator_q_value
                turn_metadata["advantage"] = coordinator_advantage
            if worker_model != model:
                turn_metadata["escalated_to"] = worker_model
        # the initial plan has no turn of its own, so attach its usage to the first worker turn
        if pending_plan_usage is not None:
            turn_metadata["plan_usage"] = pending_plan_usage
            pending_plan_usage = None

        turns.append(Turn(
            step=env_step,
            role="worker",
            obs_before=current_obs,
            action=chosen,
            obs_after=next_obs,
            env_reward=reward,
            done=done,
            metadata=turn_metadata,
        ))
        current_obs = next_obs

    return Trajectory(
        task_id=task_id,
        task_goal=task_goal,
        plan=ep_plan,
        turns=turns,
        won=env.won(info),
        total_steps=env_step,
    )
