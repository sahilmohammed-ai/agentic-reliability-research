import re

from agents.llm import complete_with_usage

# frozen LLM verifier: a single 0-1 progress score per turn from an unmodified, untrained LLM
# via prompting, no value head, no checkpoint, no gradients. distinct from verifier/infer.py's
# Verifier class, which loads a trained checkpoint (verifier v1/v2, ALFWorld-era). this is the
# "frozen LLM verifier" suggested as worth trying: score turns directly with the same model
# family (Qwen2.5-3B) used throughout the build 01-07 agentic experiments, no training pipeline.
#
# deliberately excludes env_reward/won from the prompt -- those are the ground truth this verifier
# is meant to approximate, and TextWorldExpress's real per-step reward already exists in the
# collected data, so a verifier that reads it back isn't verifying anything.
# requires a one-sentence reasoning step before the score. confirmed necessary, not cosmetic: a
# bare "output only a number" version of this prompt scored a clearly-correct, environment-
# confirmed action (following a stated instruction exactly, observation confirms success) as 0.0
# with no visible reasoning; the identical situation scored 1.0 once the model had to state its
# reasoning first. qwen2.5-3b appears to need externalized reasoning to judge a single micro-
# action correctly rather than defaulting to a low score, a known small-model prompting effect,
# not a hard capability ceiling.
SYSTEM = """\
You are a progress verifier for an agent acting in a text-based environment.
You will be given the task goal, the full sequence of actions taken so far in this episode, and
the most recent turn's observation, action, and resulting observation. Judge how good THIS ONE
TURN's action was.

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

First, in one sentence, state whether the action was correct and why. Then output a score.

Output in exactly this format:
Reasoning: <one sentence>
Score: <a single number between 0.0 and 1.0>"""


def _parse_score(raw: str) -> float:
    """extract a float in [0, 1] from the judge's "Score: <number>" line, clamping out-of-range
    values. falls back to searching the whole output if the "Score:" label is missing (e.g. the
    model dropped the format), then to 0.0 if no number is found anywhere (fail-safe: an
    unparseable judgment counts as no signal rather than crashing the caller). looking for the
    labeled line specifically, not just the first number in the text, avoids accidentally
    grabbing a number mentioned in the reasoning sentence instead of the actual score."""
    labeled = re.search(r"score:\s*(-?\d+\.?\d*)", raw, re.IGNORECASE)
    match = labeled or re.search(r"-?\d+\.?\d*", raw)
    if not match:
        return 0.0
    value = float(match.group(1) if labeled else match.group())
    return max(0.0, min(1.0, value))


def score_turn(
    task_goal: str,
    action_history: list[str],
    obs_before: str,
    action: str,
    obs_after: str,
    model: str = "hf:Qwen/Qwen2.5-3B-Instruct",
) -> tuple[float, dict]:
    """score one turn's progress with a frozen (untrained) LLM judge. returns (score, usage).

    action_history is everything taken BEFORE this turn (not including `action`), giving the
    judge the same repetition/stall context a coordinator would have."""
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
        "Reasoning and score for this turn:"
    )
    raw, usage = complete_with_usage(model, SYSTEM, prompt, max_tokens=1024)
    score = _parse_score(raw)
    usage["raw_output"] = raw
    return score, usage
