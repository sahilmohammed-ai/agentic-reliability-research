"""
single-episode runner with a fixed (heuristic) coordinator.

three coordination modes, controlled by replan_mode:
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

in all modes: thinker runs once at the start to generate a plan, worker acts every step.
"""

import uuid
from collections import Counter

from envs.alfworld_env import AlfWorldEnv
from agents import thinker, worker
from rollout.schemas import Turn, Trajectory

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
) -> Trajectory:
    """run one episode end-to-end with the fixed coordinator. returns a trajectory."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    steps_since_replan = 0  # stagnation mode's cooldown counter

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

        # worker: select and execute one action (from the possibly-masked admissible set)
        chosen, worker_usage = worker.act(task_goal, ep_plan, current_obs, worker_admissible, action_history, model=model, env_hint=env_hint)
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        steps_since_replan += 1
        action_history.append(chosen)

        turn_metadata = {"admissible_commands": admissible, "usage": worker_usage}
        if masked_action is not None:
            turn_metadata["masked_action"] = masked_action
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
