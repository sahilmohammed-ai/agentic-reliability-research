"""
single-episode runner, zero-coordination baseline.

thinker plans once at episode start; worker acts every step until done or the step cap. no
replanning, no masking, no verifier-driven intervention. this project previously had a fixed
heuristic coordinator (interval/stagnation/mask modes) and a verifier-driven "coordinator v0"
(rollout/coordinator.py, continue/retry/replan/backtrack/escalate action toolkit) -- both were
removed in favor of isolating what the Thinker/Worker models alone contribute, with zero
coordination confounds. history of that work (streak-decay fixes, the escalate-frequency
investigation, build_11/build_12's diagnosis) is kept in reports/old/build_11.md,
reports/old/build_12.md, and .info/BUILD_11_12_DIAGNOSIS.md as historical record, not reflected
in this file anymore.
"""

import uuid

from envs.alfworld_env import AlfWorldEnv
from agents import single_agent, thinker, worker
from rollout.schemas import Turn, Trajectory

# same model used for both thinker and worker in a given run, swapped per performance test
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 50        # default episode step cap (alfworld); an env may advertise its own
                      # via a step_limit attribute (scienceworld uses 100)


def run_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """run one episode end-to-end with zero coordination. returns a trajectory."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

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

        # worker: select and execute one action
        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, admissible, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turn_metadata = {"admissible_commands": admissible, "usage": worker_usage}
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


def run_single_agent_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """run one episode with a single model acting alone, no thinker plan, no coordination.
    the true single-model capability baseline: one call per step, goal+obs+admissible commands
    in, one action out. distinct from run_episode's thinker+worker agentic loop above."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    done = False
    current_obs = obs

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        chosen, usage = single_agent.act(
            task_goal, current_obs, admissible, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turns.append(Turn(
            step=env_step,
            role="worker",
            obs_before=current_obs,
            action=chosen,
            obs_after=next_obs,
            env_reward=reward,
            done=done,
            metadata={"admissible_commands": admissible, "usage": usage},
        ))
        current_obs = next_obs

    return Trajectory(
        task_id=task_id,
        task_goal=task_goal,
        plan="",  # no thinker plan in the single-agent baseline
        turns=turns,
        won=env.won(info),
        total_steps=env_step,
    )
