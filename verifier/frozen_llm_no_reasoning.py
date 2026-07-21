import re

from agents.llm import complete_with_usage

# variant of verifier/frozen_llm.py that drops the required reasoning step: bare "output only a
# number" prompt, same judging criteria otherwise. build 08 rejected this shape after ONE example
# (a correct, environment-confirmed action scored 0.0 with no visible reasoning) and switched to
# requiring reasoning before the score. this module exists to re-test that call systematically,
# at scale, against build 03's full 100-episode baseline, rather than trust a single example --
# see reports/twx/09_verifier_variants.md for the result. NOT used by any live coordinator;
# offline-scoring only, same as verifier/frozen_llm.py's evaluation use in build 08.
SYSTEM = """\
You are a progress verifier for an agent acting in a text-based environment.
You will be given the task goal, the full sequence of actions taken so far in this episode, and
the most recent turn's observation, action, and resulting observation. Judge how good THIS ONE
TURN's action was, on a scale from 0.0 to 1.0.

Judge based on the action itself, not on whether the whole task is finished yet -- most turns in
a multi-turn task are correct steps toward the goal without completing it, and those should score
high too, not just the final turn. Ask: "given everything tried so far and the observation
BEFORE this action, was this action the right thing to do at that moment?"

Important: judge the action only against what was known BEFORE it was taken (the "Observation
before" text). The "Observation after" text may contain a new instruction or new information that
only appeared as a RESULT of this action -- that new information was not available when the
action was chosen, so never judge the action as wrong for failing to anticipate it.

0.0 means the action was wrong, repeated something already tried with no new result, or moved
away from the goal. 0.5 means the action was reasonable but uncertain or exploratory (e.g. a
sensible guess when the right move genuinely isn't clear yet). 1.0 means the action was clearly
correct given the situation -- this includes a normal, on-track step in a longer sequence, not
only the final action that completes the task.

Output only a single number between 0.0 and 1.0, nothing else."""


def _parse_score(raw: str) -> float:
    """same fail-safe parsing as frozen_llm.py, but simpler since there's no reasoning text to
    accidentally match a number inside of -- just take the first number found."""
    match = re.search(r"-?\d+\.?\d*", raw)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


def score_turn(
    task_goal: str,
    action_history: list[str],
    obs_before: str,
    action: str,
    obs_after: str,
    model: str = "hf:Qwen/Qwen2.5-3B-Instruct",
) -> tuple[float, dict]:
    """same signature/contract as verifier/frozen_llm.py's score_turn, no-reasoning variant."""
    history_block = (
        "\n".join(f"- {a}" for a in action_history) if action_history else "(no actions yet)"
    )
    prompt = (
        f"Task: {task_goal}\n\n"
        f"Actions taken so far (before this turn):\n{history_block}\n\n"
        f"This turn:\n"
        f"Observation before: {obs_before}\n"
        f"Action taken: {action}\n"
        f"Observation after: {obs_after}\n\n"
        "Score for this turn (0.0 to 1.0):"
    )
    raw, usage = complete_with_usage(model, SYSTEM, prompt, max_tokens=64)
    score = _parse_score(raw)
    usage["raw_output"] = raw
    return score, usage
