"""
single-episode runner with a fixed (heuristic) coordinator.

fixed coordination policy:
  - thinker runs once at the start to generate a plan
  - worker acts every step, selecting from admissible commands

this fixed coordinator is baseline 1 in the evaluation.
"""

import uuid
from envs.alfworld_env import AlfWorldEnv
from agents import thinker, worker
from rollout.schemas import Turn, Trajectory

# same model used for both thinker and worker in a given run, swapped per performance test
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 50  # episode step cap


def run_episode(env: AlfWorldEnv, task_id: str | None = None, model: str = DEFAULT_MODEL) -> Trajectory:
    """run one episode end-to-end with the fixed coordinator. returns a trajectory."""
    task_id = task_id or str(uuid.uuid4())[:8]
    turns: list[Turn] = []
    action_history: list[str] = []
    env_step = 0

    obs, info = env.reset()
    # pull the real task line, not just the banner (obs[0] was the old, wrong extraction)
    obs_lines = [l.strip() for l in obs.split("\n")]
    task_goal = next((l for l in obs_lines if l.startswith("Your task is to:")), obs_lines[0])

    # thinker: generate plan once at episode start
    ep_plan = thinker.plan(task_goal, obs, model=model)

    done = False
    current_obs = obs

    while not done and env_step < MAX_STEPS:
        admissible = env.admissible_commands(info)

        # worker: select and execute one action
        chosen = worker.act(task_goal, ep_plan, current_obs, admissible, action_history, model=model)
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
