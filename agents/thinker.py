from agents.llm import complete

# thinker generates high-level plan from task goal and initial observation
SYSTEM = """\
You are a planning agent inside a household task system.
Given a task goal and the current scene description, produce a concise numbered plan.
Each step should be a single, concrete physical action (e.g. "1. go to the desk").
Output only the numbered plan, nothing else."""


def plan(task_goal: str, initial_obs: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """generate high-level plan from task goal and initial scene."""
    # build prompt with task and initial observation
    prompt = f"Task: {task_goal}\n\nScene:\n{initial_obs}\n\nPlan:"
    return complete(model, SYSTEM, prompt, max_tokens=256)
