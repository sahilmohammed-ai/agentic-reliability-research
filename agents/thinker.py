from agents.llm import complete_with_usage

# thinker generates high-level plan from task goal and initial observation.
#
# requirements-first format: a task goal can bundle multiple requirements beyond "move object X
# to location Y" (e.g. "put a HOT mug in cabinet" also requires a state change, "put TWO cd in
# drawer" requires two separate objects). found via direct data audit (build_10_alfworld, 216
# episodes whose goal needed heat/cool/clean): a plan generated from a bare "produce a plan"
# instruction omitted the required verb entirely 51.9% of the time, and 88% of those omissions
# ended in a loss, even though the initial observation always listed the object needed to satisfy
# it (e.g. a microwave/sink was visible). the model had the information and dropped the
# requirement anyway under an unstructured prompt.
#
# a first fix (instruction-only: "first identify every requirement...") was NOT sufficient on its
# own. tested directly against the real failure case ("put a hot mug in cabinet") with
# hf:Qwen/Qwen2.5-3B-Instruct: the model still produced "Requirements: find a mug, take a mug to
# cabinet", dropping "hot" even when explicitly told to enumerate state-change requirements,
# confirmed deterministic (greedy decoding) across repeated calls, not a fluke.
#
# a second fix (one few-shot example, cool/fridge) was PARTIAL: re-tested against the same real
# case, the requirements line correctly said "find a HOT mug" this time, and the plan correctly
# went to the microwave instead of the coffeemachine -- but still never emitted an explicit "heat
# mug with microwave 1" action, just "take mug from microwave 1" then straight to the cabinet.
#
# a THIRD fix (two few-shot examples, cool AND clean, with an explicit callout: "Notice step 4 in
# both examples... do not skip it") made NO further difference: re-tested a third time against the
# identical real case, byte-for-byte the same plan as the one-example version -- requirements line
# correct ("find a hot mug"), goes to the right appliance, still never performs the heat action.
#
# CONCLUSION: this is a genuine Qwen2.5-3B-Instruct capability ceiling on this specific
# compositional pattern (object + state-change-via-appliance + destination), not a prompt-wording
# problem. Two clear worked examples plus an explicit "don't skip this step" callout did not move
# the model off treating "pick up the object near the right appliance" as equivalent to "perform
# the state-change action on it." Stopped iterating on the prompt here per that evidence; see
# .info/CLAUDE.md for the documented finding. The partial improvement (correct requirements
# line, correct appliance choice) is kept since it's a real, verified improvement over the
# original total omission, even though it doesn't fully solve the task.
SYSTEM = """\
You are a planning agent inside a household task system.
Given a task goal and the current scene description, first identify every requirement the goal \
implies, then produce a concise numbered plan that satisfies all of them.

A requirement is anything the goal states beyond "move an object to a place": a state change \
(e.g. hot/cold/clean), a quantity (e.g. "two" of something), or a specific supporting object \
(e.g. "examine X with the desklamp" requires finding and using the desklamp). A state-change \
requirement is NEVER satisfied just by picking the object up near the right appliance -- it \
always needs its own explicit action step performed on the object.

Example 1:
Task: Your task is to: put a cool tomato in the fridge.
Scene: You see a fridge 1, a countertop 1, a garbagecan 1.
Requirements: find a tomato, cool the tomato (state change), put it in the fridge
Plan:
1. go to countertop 1
2. take tomato from countertop 1
3. go to fridge 1
4. cool tomato with fridge 1
5. put tomato in/on fridge 1

Example 2:
Task: Your task is to: put a clean plate on the countertop.
Scene: You see a sinkbasin 1, a countertop 1, a diningtable 1.
Requirements: find a plate, clean the plate (state change), put it on the countertop
Plan:
1. go to diningtable 1
2. take plate from diningtable 1
3. go to sinkbasin 1
4. clean plate with sinkbasin 1
5. go to countertop 1
6. put plate in/on countertop 1

Notice step 4 in both examples: a dedicated action ("cool X with Y", "clean X with Y") performs \
the state change. Picking the object up near the appliance is NOT that step; do not skip it.

Now do the same for the real task below. Output in exactly this format:
Requirements: <comma-separated list of every requirement implied by the goal>
Plan:
1. <first concrete physical action>
2. <second concrete physical action>
...

Each plan step should be a single, concrete physical action (e.g. "1. go to the desk"). Output \
only the Requirements line and the numbered plan, nothing else."""

# replan sees what was already tried, so it must actively revise rather than repeat
REPLAN_SYSTEM = """\
You are a planning agent inside a household task system.
Your previous plan is not working. You will be given the task goal, your old plan, the actions
taken so far, and the current scene. The actions taken so far show what has already been tried,
including any repeated or stuck behavior.

First re-identify every requirement the goal implies (state changes like hot/cold/clean,
quantities like "two", specific supporting objects), since a plan that stalls partway through may
have already dropped one. Then produce a new concise numbered plan that satisfies all requirements
and avoids repeating what clearly is not working.

Output in exactly this format:
Requirements: <comma-separated list of every requirement implied by the goal>
Plan:
1. <first concrete physical action>
2. <second concrete physical action>
...

Each plan step should be a single, concrete physical action. Output only the Requirements line and
the numbered plan, nothing else."""


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
