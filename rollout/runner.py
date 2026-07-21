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
from verifier.frozen_llm import score_turn as verifier_score_turn

# same model used for both thinker and worker in a given run, swapped per performance test
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 50        # default episode step cap (alfworld); an env may advertise its own
                      # via a step_limit attribute (scienceworld uses 100)

REPEAT_THRESHOLD = 3   # N identical consecutive actions triggers a replan (fixed coordinator v1)
CYCLE_MIN_PERIOD = 2   # shortest repeating action sequence to detect (e.g. A,B,A,B)
CYCLE_MAX_PERIOD = 4   # longest repeating action sequence to detect
CYCLE_MIN_REPEATS = 2  # how many times the period must repeat back-to-back to count as a cycle

# reward-aware coordinator (v3): triggers on env_reward staying non-positive for a run of steps,
# instead of a string-pattern match on actions. threshold picked from direct data audit of build
# 03's no-coordinator baseline: non-positive-reward streak length cleanly separates won vs lost
# episodes on coin (won avg 14.5 steps, lost avg 50.0) and mapreader (won avg 9.4, lost avg 45.5),
# while simonsays/peckingorder (dense per-step reward) almost never build a long streak at all
# (avg 0.0 and 1.2) -- so a single threshold works across the game mix without needing to be
# scaled per-game. 15 sits comfortably above the won-episode averages (~10-15) and well below the
# lost-episode averages (~45-50), so it should rarely fire on episodes that are actually
# progressing.
REWARD_STALL_THRESHOLD = 15   # consecutive non-positive-reward steps before the first mask fires
ESCALATE_AFTER_STEPS = 5      # additional non-positive-reward steps after the mask before replanning

# verifier-triggered coordinator (v5): uses the frozen LLM verifier's per-turn score
# (verifier/frozen_llm.py) instead of env_reward or a string-pattern match as the stall signal.
# threshold/streak picked to mirror build 06's reward-stall design (streak-based, not
# single-turn) while being much shorter, since the verifier gives a real per-turn judgment
# rather than an accumulated reward that's naturally sparse -- 5 consecutive clearly-bad
# judgments (score < 0.3) is already a strong signal, unlike reward which needs ~15 steps to
# separate signal from normal sparse-reward exploration (see build 08's report for the
# score/won-lost correlation this threshold is built on).
VERIFIER_SCORE_THRESHOLD = 0.3
VERIFIER_STALL_STREAK = 5


def _is_repeating(history: list[str], threshold: int = REPEAT_THRESHOLD) -> bool:
    """true if the last `threshold` actions are all identical (e.g. A,A,A)."""
    if len(history) < threshold:
        return False
    return len(set(history[-threshold:])) == 1


def _is_cycling(
    history: list[str],
    min_period: int = CYCLE_MIN_PERIOD,
    max_period: int = CYCLE_MAX_PERIOD,
    min_repeats: int = CYCLE_MIN_REPEATS,
) -> bool:
    """true if the tail of history is a short sequence repeating back-to-back at least
    `min_repeats` times (e.g. period 2: A,B,A,B; period 4: A,B,C,D,A,B,C,D). catches loops that
    aren't a single repeated action (see _is_repeating) but cycle through a small fixed set in a
    fixed order. does NOT catch semantically-stuck-but-lexically-varied churn (confirmed via a
    real cookingworld trace: 6/6 distinct actions in most 6-action windows despite the episode
    never making progress) -- that class of stall needs reward or verifier signal, not a
    string-pattern heuristic, and is out of scope for this fixed coordinator."""
    for period in range(min_period, max_period + 1):
        window = period * min_repeats
        if len(history) < window:
            continue
        recent = history[-window:]
        chunks = [tuple(recent[i:i + period]) for i in range(0, window, period)]
        if len(set(chunks)) == 1:
            return True
    return False


def _looping_actions(history: list[str]) -> set[str]:
    """returns the distinct action(s) involved in a detected repeat or cycle at the tail of
    history, or an empty set if neither _is_repeating nor _is_cycling fires. used by the masking
    coordinator to know what to hide from the worker's next choice; the replan coordinator doesn't
    need this since it hands the whole history to thinker.replan() instead."""
    if _is_repeating(history):
        return set(history[-REPEAT_THRESHOLD:])
    for period in range(CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD + 1):
        window = period * CYCLE_MIN_REPEATS
        if len(history) < window:
            continue
        recent = history[-window:]
        chunks = [tuple(recent[i:i + period]) for i in range(0, window, period)]
        if len(set(chunks)) == 1:
            return set(recent)
    return set()


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


def run_coordinated_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """thinker+worker loop with one fixed coordinator action: replan when the worker is stuck in
    a literal repetition (_is_repeating) or a short cycle (_is_cycling). this is the first step
    toward a learned coordinator -- simple, observable string-pattern triggers (no reward/verifier
    signal needed) to validate the mechanism before anything is learned. distinct from run_episode
    (zero coordination) and run_single_agent_episode (no thinker at all)."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        # fixed coordinator: replan when stuck in a literal repetition or a short cycle. reset
        # history to just the last action after replanning so the same stall can't retrigger a
        # replan on every subsequent step.
        replanned = False
        if _is_repeating(action_history) or _is_cycling(action_history):
            ep_plan, replan_usage = thinker.replan(
                task_goal, ep_plan, action_history, current_obs, model=model,
            )
            replanned = True
            pending_plan_usage = replan_usage
            action_history = action_history[-1:]  # keep last action for worker context, drop the rest of the streak

        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, admissible, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turn_metadata = {"admissible_commands": admissible, "usage": worker_usage, "replanned": replanned}
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


def run_masked_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """thinker+worker loop with a fixed coordinator action: on the same trigger as
    run_coordinated_episode (literal repeat or short cycle), instead of replanning, mask the
    looping action(s) out of the admissible list handed to the worker so it physically cannot
    repeat the loop and must pick something else. isolates the masking intervention from
    replanning for a direct comparison on the same detector/trigger. no thinker involvement beyond
    the initial plan -- this coordinator never calls replan()."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        # fixed coordinator: mask the looping action(s) out of what the worker can choose. if
        # masking would remove every admissible option, skip masking for this turn rather than
        # hand the worker an empty choice -- letting it act (even into the loop again) is better
        # than a crash or an undefined fallback.
        looping = _looping_actions(action_history)
        masked = [c for c in admissible if c not in looping]
        worker_choices = masked if masked else admissible

        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, worker_choices, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        turn_metadata = {
            "admissible_commands": admissible,
            "usage": worker_usage,
            "masked_actions": sorted(looping),
        }
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


def run_reward_aware_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """thinker+worker loop with a reward-aware fixed coordinator (v3), replacing builds 04/05's
    string-pattern detectors (_is_repeating/_is_cycling) entirely. tracks a running streak of
    consecutive non-positive env_reward steps:
      - at REWARD_STALL_THRESHOLD, mask the single most recent action (cheap first response).
      - if ESCALATE_AFTER_STEPS more steps pass with still-non-positive reward, escalate to a
        full thinker.replan() and reset both counters.
      - any positive-reward step resets the streak (and the escalation sub-counter) to 0.
    unlike the string detectors, this can catch semantically-stuck-but-lexically-varied churn
    (e.g. cookingworld's real failure mode) since it only looks at reward, not action identity."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    stall_streak = 0        # consecutive non-positive-reward steps since the last reset
    steps_since_mask = 0    # non-positive-reward steps since the first mask fired, or -1 if not yet in a masked stall

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage
    in_masked_stall = False

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        replanned = False
        masked_actions: set[str] = set()

        if in_masked_stall and steps_since_mask >= ESCALATE_AFTER_STEPS:
            # masking didn't resolve the stall in time -- escalate to a full replan.
            ep_plan, replan_usage = thinker.replan(
                task_goal, ep_plan, action_history, current_obs, model=model,
            )
            replanned = True
            pending_plan_usage = replan_usage
            stall_streak = 0
            steps_since_mask = 0
            in_masked_stall = False
        elif stall_streak >= REWARD_STALL_THRESHOLD and action_history:
            # first response to a sustained non-positive-reward stall: mask the last action.
            masked_actions = {action_history[-1]}
            in_masked_stall = True

        worker_choices = [c for c in admissible if c not in masked_actions] or admissible

        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, worker_choices, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        if reward > 0:
            stall_streak = 0
            steps_since_mask = 0
            in_masked_stall = False
        else:
            stall_streak += 1
            if in_masked_stall:
                steps_since_mask += 1

        turn_metadata = {
            "admissible_commands": admissible,
            "usage": worker_usage,
            "replanned": replanned,
            "masked_actions": sorted(masked_actions),
            "stall_streak": stall_streak,
        }
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


def run_backtrack_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """thinker+worker loop with one fixed coordinator action: backtrack. same trigger as
    run_reward_aware_episode (REWARD_STALL_THRESHOLD consecutive non-positive-reward steps), but
    the response is neither replanning nor masking -- it clears action_history entirely, so the
    worker's next choice sees only the current observation and the original plan, no memory of
    the stalled recent context. no escalation path; backtrack is the only action tested here, so
    a stall that isn't resolved by one backtrack can trigger another once the streak rebuilds to
    threshold. isolates a third, genuinely different intervention type from build 04's replan and
    build 05's mask: neither narrows choices nor gets new strategy, just clears stale context."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0
    stall_streak = 0

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        backtracked = False
        if stall_streak >= REWARD_STALL_THRESHOLD:
            action_history = []
            backtracked = True
            stall_streak = 0

        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, admissible, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        stall_streak = 0 if reward > 0 else stall_streak + 1

        turn_metadata = {
            "admissible_commands": admissible,
            "usage": worker_usage,
            "backtracked": backtracked,
            "stall_streak": stall_streak,
        }
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


VERIFIER_INTERVENTIONS = ("mask", "replan", "backtrack")


def run_verifier_coordinated_episode(
    env: AlfWorldEnv,
    task_id: str | None = None,
    model: str = DEFAULT_MODEL,
    intervention: str = "mask",
) -> Trajectory:
    """thinker+worker loop with a coordinator triggered by the frozen LLM verifier's per-turn
    score (verifier/frozen_llm.py), instead of a string-pattern match or raw env_reward. after
    each turn, the verifier scores it (never sees env_reward/won); if the last VERIFIER_STALL_
    STREAK scores are all below VERIFIER_SCORE_THRESHOLD, the coordinator intervenes before the
    NEXT action is chosen. `intervention` selects which response fires on trigger:
      - "mask": remove the most recent action from the next turn's admissible list.
      - "replan": call thinker.replan() with the full action history.
      - "backtrack": clear action_history entirely.
    all three share the same trigger so the comparison isolates intervention type, mirroring how
    builds 04/05 isolated intervention type on the same string-pattern trigger. no escalation --
    the streak resets after any intervention fires."""
    if intervention not in VERIFIER_INTERVENTIONS:
        raise ValueError(f"intervention must be one of {VERIFIER_INTERVENTIONS}, got {intervention!r}")

    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    verifier_scores: list[float] = []
    env_step = 0

    max_steps = getattr(env, "step_limit", MAX_STEPS)

    obs, info = env.reset()
    task_goal = env.task_goal(obs, info)
    env_hint = env.worker_hint() if hasattr(env, "worker_hint") else ""

    ep_plan, initial_plan_usage = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs
    pending_plan_usage = initial_plan_usage

    while not done and env_step < max_steps:
        admissible = env.admissible_commands(info)

        # coordinator: intervene if the verifier has judged the last VERIFIER_STALL_STREAK turns
        # as all below threshold. streak resets after any intervention fires.
        triggered = (
            len(verifier_scores) >= VERIFIER_STALL_STREAK
            and all(s < VERIFIER_SCORE_THRESHOLD for s in verifier_scores[-VERIFIER_STALL_STREAK:])
        )
        intervened = False
        masked_actions: set[str] = set()

        if triggered and action_history:
            intervened = True
            if intervention == "mask":
                masked_actions = {action_history[-1]}
            elif intervention == "replan":
                ep_plan, replan_usage = thinker.replan(
                    task_goal, ep_plan, action_history, current_obs, model=model,
                )
                pending_plan_usage = replan_usage
            elif intervention == "backtrack":
                action_history = []
            verifier_scores = []

        worker_choices = [c for c in admissible if c not in masked_actions] or admissible

        chosen, worker_usage = worker.act(
            task_goal, ep_plan, current_obs, worker_choices, action_history,
            model=model, env_hint=env_hint,
        )
        next_obs, reward, done, info = env.step(chosen)
        env_step += 1
        action_history.append(chosen)

        v_score, v_usage = verifier_score_turn(
            task_goal=task_goal,
            action_history=action_history[:-1],
            obs_before=current_obs,
            action=chosen,
            obs_after=next_obs,
            model=model,
        )
        verifier_scores.append(v_score)

        turn_metadata = {
            "admissible_commands": admissible,
            "usage": worker_usage,
            "verifier_score": v_score,
            "verifier_usage": v_usage,
            "intervened": intervened,
            "intervention_type": intervention if intervened else None,
            "masked_actions": sorted(masked_actions),
        }
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
