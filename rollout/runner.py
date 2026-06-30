"""
Single-episode runner with a fixed (heuristic) coordinator.

Fixed coordination policy:
  - Thinker runs once at the start to generate a plan.
  - Worker acts every step, selecting from admissible commands.
  - Verifier-role checks every VERIFIER_INTERVAL steps (advisory; result stored in turn metadata).

This fixed coordinator is Baseline 1 in the evaluation. The learned PPO coordinator will replace
the heuristic routing logic here once the verifier is trained.
"""

import uuid
from envs.alfworld_env import AlfWorldEnv
from agents import thinker, worker, verifier_role
from rollout.schemas import Turn, Trajectory

VERIFIER_INTERVAL = 3   # how often the verifier_role checks (in env steps)
MAX_STEPS = 50          # episode step cap


def run_episode(env: AlfWorldEnv, task_id: str | None = None) -> Trajectory:
    """Run one episode end-to-end with the fixed coordinator. Returns a Trajectory."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

    obs, info = env.reset()
    task_goal = obs.split("\n")[0].strip()

    # Thinker: generate plan once at episode start.
    ep_plan = thinker.plan(task_goal, obs)

    done = False
    current_obs = obs

    while not done and env_step < MAX_STEPS:
        admissible = env.admissible_commands(info)

        # Verifier-role: advisory check every VERIFIER_INTERVAL steps.
        if env_step > 0 and env_step % VERIFIER_INTERVAL == 0:
            check_result = verifier_role.check(task_goal, ep_plan, action_history, current_obs)
            turns.append(Turn(
                step=env_step,
                role="verifier_role",
                obs_before=current_obs,
                action=check_result,
                obs_after="",
                env_reward=0.0,
                done=False,
                metadata={"type": "advisory_check"},
            ))

        # Worker: select and execute one action.
        chosen = worker.act(task_goal, ep_plan, current_obs, admissible, action_history)
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
            metadata={"admissible_commands": admissible},
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
