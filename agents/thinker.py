from agents.llm import complete_with_usage

# thinker generates high-level plan from task goal and initial observation.
# it is shown the environment's actual admissible commands so it plans in the real action
# vocabulary rather than inventing commands the environment doesn't accept (build_9 exposed
# this: the thinker produced "pick up the instructions book" when the only valid command was
# "read instructions book", and the worker lost in one step following the impossible plan).
SYSTEM = """\
You are a planning agent inside a household task system.
Given a task goal, the current scene description, and the list of commands the environment
actually accepts, produce a concise numbered plan.
Each plan step must be phrased using the environment's real command vocabulary shown to you,
not invented actions. Output only the numbered plan, nothing else."""

# replan sees what was already tried, so it must actively revise rather than repeat
REPLAN_SYSTEM = """\
You are a planning agent inside a household task system.
Your previous plan is not working. You will be given the task goal, your old plan, the actions
taken so far, the current scene, and the list of commands the environment actually accepts.
The actions taken so far show what has already been tried, including any repeated or stuck
behavior. Produce a new concise numbered plan that avoids repeating what clearly is not working.
Each plan step must be phrased using the environment's real command vocabulary shown to you,
not invented actions. Output only the numbered plan, nothing else."""


def _commands_block(admissible_commands: list[str] | None) -> str:
    """render the available-commands section, empty if none supplied (keeps old callers working)."""
    if not admissible_commands:
        return ""
    listed = "\n".join(f"- {c}" for c in admissible_commands)
    return f"Commands the environment accepts right now:\n{listed}\n\n"


def plan(
    task_goal: str,
    initial_obs: str,
    admissible_commands: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[str, dict]:
    """generate high-level plan from task goal, initial scene, and the real command vocabulary.
    returns (plan, usage)."""
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Scene:\n{initial_obs}\n\n"
        f"{_commands_block(admissible_commands)}"
        "Plan:"
    )
    return complete_with_usage(model, SYSTEM, prompt, max_tokens=256)


def replan(
    task_goal: str,
    old_plan: str,
    history: list[str],
    current_obs: str,
    admissible_commands: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[str, dict]:
    """revise the plan mid-episode, given the full action history and the real command
    vocabulary. returns (plan, usage)."""
    history_block = "\n".join(f"- {a}" for a in history) if history else "(no actions yet)"
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Old plan:\n{old_plan}\n\n"
        f"Actions taken so far:\n{history_block}\n\n"
        f"Current scene:\n{current_obs}\n\n"
        f"{_commands_block(admissible_commands)}"
        "New plan:"
    )
    return complete_with_usage(model, REPLAN_SYSTEM, prompt, max_tokens=256)
