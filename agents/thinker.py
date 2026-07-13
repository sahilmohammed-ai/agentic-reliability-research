from agents.llm import complete_with_usage

# thinker generates high-level plan from task goal and initial observation
SYSTEM = """\
You are a planning agent inside a household task system.
Given a task goal and the current scene description, produce a concise numbered plan.
Each step should be a single, concrete physical action (e.g. "1. go to the desk").
Output only the numbered plan, nothing else."""

# replan sees what was already tried, so it must actively revise rather than repeat
REPLAN_SYSTEM = """\
You are a planning agent inside a household task system.
Your previous plan is not working. You will be given the task goal, your old plan, the actions
taken so far, and the current scene. The actions taken so far show what has already been tried,
including any repeated or stuck behavior. Produce a new concise numbered plan that avoids
repeating what clearly is not working. Each step should be a single, concrete physical action.
Output only the numbered plan, nothing else."""


def plan(task_goal: str, initial_obs: str, model: str = "claude-haiku-4-5-20251001") -> tuple[str, dict]:
    """generate high-level plan from task goal and initial scene. returns (plan, usage)."""
    # build prompt with task and initial observation
    prompt = f"Task: {task_goal}\n\nScene:\n{initial_obs}\n\nPlan:"
    return complete_with_usage(model, SYSTEM, prompt, max_tokens=256)


def replan(
    task_goal: str,
    old_plan: str,
    history: list[str],
    current_obs: str,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[str, dict]:
    """revise the plan mid-episode, given the full action history so far. returns (plan, usage)."""
    history_block = "\n".join(f"- {a}" for a in history) if history else "(no actions yet)"
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Old plan:\n{old_plan}\n\n"
        f"Actions taken so far:\n{history_block}\n\n"
        f"Current scene:\n{current_obs}\n\n"
        "New plan:"
    )
    return complete_with_usage(model, REPLAN_SYSTEM, prompt, max_tokens=256)
