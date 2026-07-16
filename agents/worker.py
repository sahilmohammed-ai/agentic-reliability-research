from agents.llm import complete_with_usage

# worker picks the next action to take from available commands.
#
# direct data audit (build_10_alfworld, 21,399 real worker turns) found a strong, likely-genuine
# behavioral bias: "examine" is 39.2% of every action ever taken, with runs of 5+ consecutive
# examine calls occurring 231 times (worst case: 49 in a row, essentially an entire episode spent
# examining). a small model defaults to a low-commitment "look at something" action when uncertain
# rather than committing to a step of the plan. the old prompt gave no guidance against this.
# added: an explicit instruction to prefer the next unfinished plan step over "examine" unless
# examine is the plan step itself, and a rule against repeating the same command twice in a row.
SYSTEM = """\
You are an action agent inside a household task system.
You will be given the task goal, a high-level plan, the current scene observation, and a list of
admissible commands. Choose exactly one command from the admissible list that best advances the \
next unfinished step of the plan.

Do not choose "examine" or "look" commands unless the plan step itself requires examining \
something, or you genuinely cannot tell which admissible command matches the plan's next step. \
Prefer committing to a concrete action (go to / take / open / put / heat / cool / clean / move) \
over observing. Do not repeat the exact same command you just took unless the observation shows \
it produced a new, different result.

Output only the chosen command, copied exactly as it appears in the list."""


def act(
    task_goal: str,
    plan: str,
    obs: str,
    admissible_commands: list[str],
    history: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    env_hint: str = "",
) -> tuple[str, dict]:
    """select one admissible command that best advances the plan.

    returns (action, usage) where usage is the token counts for this call, so the runner can
    log per-turn cost. env_hint is optional environment-specific guidance injected into the
    prompt (e.g. a warning about scienceworld's irreversible 'focus on' action). empty for
    alfworld, so the two environments stay comparable except where an env genuinely needs a caveat."""
    # optionally include last few actions for context
    history_block = ""
    if history:
        history_block = "Recent actions taken:\n" + "\n".join(f"- {a}" for a in history[-5:]) + "\n\n"

    hint_block = f"{env_hint}\n\n" if env_hint else ""

    # format available commands into list
    commands_block = "\n".join(f"- {c}" for c in admissible_commands)
    # build full decision prompt with context
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{hint_block}"
        f"{history_block}"
        f"Current observation:\n{obs}\n\n"
        f"Admissible commands:\n{commands_block}\n\n"
        "Choose one command from the list above:"
    )
    raw, usage = complete_with_usage(model, SYSTEM, prompt, max_tokens=64)

    # match model's output format to actual command. previously the raw output and whether a
    # fallback fired were both discarded, so it was impossible to tell "the model genuinely chose
    # this" from "the model's output didn't match anything and this is just admissible[0]" after
    # the fact. both are now recorded in usage (already a dict that flows into turn metadata,
    # rollout/runner.py:233), so this stops being an invisible failure mode in collected data.
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
