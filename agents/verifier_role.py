import anthropic

# singleton client to avoid reinitializing on each call
_client = None


def _get_client() -> anthropic.Anthropic:
    """lazy init and cache the anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# verifier checks if the agent is on track or has derailed
SYSTEM = """\
You are a verification agent inside a household task system.
Given the task goal, the plan, and the recent action history, assess whether the agent is making
progress toward the goal or has gone off track.
Respond with one of:
  OK - progress looks fine
  CONCERN: <one sentence explaining what looks wrong>
Output only the status line, nothing else."""


def check(
    task_goal: str,
    plan: str,
    history: list[str],
    latest_obs: str,
) -> str:
    """check if current trajectory is on track. returns 'ok' or 'concern: ...'."""
    # format action history into readable list
    history_block = "\n".join(f"- {a}" for a in history) if history else "(no actions yet)"
    # build full verification prompt with context
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Actions taken so far:\n{history_block}\n\n"
        f"Current observation:\n{latest_obs}\n\n"
        "Status:"
    )
    # call claude to verify trajectory
    message = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
