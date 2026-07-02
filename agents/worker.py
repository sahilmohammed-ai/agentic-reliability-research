import anthropic

# singleton client to avoid reinitializing on each call
_client = None


def _get_client() -> anthropic.Anthropic:
    """lazy init and cache the anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# worker picks the next action to take from available commands
SYSTEM = """\
You are an action agent inside a household task system.
You will be given the task goal, a high-level plan, the current scene observation, and a list of
admissible commands. Choose exactly one command from the admissible list that best advances the plan.
Output only the chosen command, copied exactly as it appears in the list."""


def act(
    task_goal: str,
    plan: str,
    obs: str,
    admissible_commands: list[str],
    history: list[str] | None = None,
) -> str:
    """select one admissible command that best advances the plan."""
    # optionally include last few actions for context
    history_block = ""
    if history:
        history_block = "Recent actions taken:\n" + "\n".join(f"- {a}" for a in history[-5:]) + "\n\n"

    # format available commands into list
    commands_block = "\n".join(f"- {c}" for c in admissible_commands)
    # build full decision prompt with context
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{history_block}"
        f"Current observation:\n{obs}\n\n"
        f"Admissible commands:\n{commands_block}\n\n"
        "Choose one command from the list above:"
    )
    # call claude to pick next action
    message = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # match claude's output format to actual command with fallback to first command
    raw_lower = raw.lower()
    for cmd in admissible_commands:
        if cmd.lower() == raw_lower:
            return cmd
    for cmd in admissible_commands:
        if cmd.lower() in raw_lower or raw_lower in cmd.lower():
            return cmd
    return admissible_commands[0]
