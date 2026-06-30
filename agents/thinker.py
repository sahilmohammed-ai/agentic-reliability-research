import anthropic

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


SYSTEM = """\
You are a planning agent inside a household task system.
Given a task goal and the current scene description, produce a concise numbered plan.
Each step should be a single, concrete physical action (e.g. "1. go to the desk").
Output only the numbered plan, nothing else."""


def plan(task_goal: str, initial_obs: str) -> str:
    """Generate a high-level plan for the episode given the task goal and first observation."""
    prompt = f"Task: {task_goal}\n\nScene:\n{initial_obs}\n\nPlan:"
    message = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
