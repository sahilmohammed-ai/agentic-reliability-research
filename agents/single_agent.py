from agents.llm import complete_with_usage

# single-agent baseline: one model sees the task goal, current observation, and admissible
# commands each step and picks an action directly. no separate thinker plan -- this is the
# true single-model capability baseline, distinct from the thinker+worker agentic loop
# (agents/thinker.py + agents/worker.py) used in build 02.
SYSTEM = """\
You are an agent completing a task in a text-based environment.
You will be given the task goal, the current scene observation, and a list of admissible
commands. Choose exactly one command from the admissible list that best advances the task.

Do not choose "examine" or "look" commands unless you genuinely cannot tell what to do next.
Prefer committing to a concrete action over observing. Do not repeat the exact same command you
just took unless the observation shows it produced a new, different result.

Output only the chosen command, copied exactly as it appears in the list."""


def act(
    task_goal: str,
    obs: str,
    admissible_commands: list[str],
    history: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    env_hint: str = "",
) -> tuple[str, dict]:
    """select one admissible command that best advances the task, no plan involved.

    returns (action, usage), same contract as agents/worker.py's act()."""
    history_block = ""
    if history:
        history_block = "Recent actions taken:\n" + "\n".join(f"- {a}" for a in history[-5:]) + "\n\n"

    hint_block = f"{env_hint}\n\n" if env_hint else ""

    commands_block = "\n".join(f"- {c}" for c in admissible_commands)
    prompt = (
        f"Task: {task_goal}\n\n"
        f"{hint_block}"
        f"{history_block}"
        f"Current observation:\n{obs}\n\n"
        f"Admissible commands:\n{commands_block}\n\n"
        "Choose one command from the list above:"
    )
    raw, usage = complete_with_usage(model, SYSTEM, prompt, max_tokens=1024)

    usage["raw_output"] = raw
    raw_lower = raw.lower()
    for cmd in admissible_commands:
        if cmd.lower() == raw_lower:
            usage["parse_fallback"] = False
            return cmd, usage
    for cmd in admissible_commands:
        if cmd.lower() in raw_lower or raw_lower in cmd.lower():
            usage["parse_fallback"] = False
            return cmd, usage
    usage["parse_fallback"] = True
    return admissible_commands[0], usage
