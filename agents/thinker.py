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
You are a planning agent that produces a short strategy note before an execution agent starts
acting in a text-based environment. You do not know in advance what kind of task this is, so
first classify it, then plan accordingly.

Three broad kinds of task exist:
1. FIXED-GOAL tasks: the goal is set once and doesn't change (e.g. "find object X and put it in
   place Y", possibly with extra requirements like a state change, a quantity, or a supporting
   object). These reward a concrete, ordered, multi-step plan made up front.
2. REACTIVE/INSTRUCTION-FOLLOWING tasks: the goal text itself says to follow instructions that
   will be revealed over time, one at a time, forever (e.g. "do exactly what Simon says", "read
   and follow the instructions book, repeat this until done"). The correct action on turn 5
   depends entirely on what the environment says on turn 5, not on anything decided now. A rigid
   step-by-step plan is actively harmful here. If you detect this kind of task, the plan should
   say so explicitly and instruct the executor to always follow the most recent instruction, not
   a pre-set sequence.
3. DISCOVER-THEN-ACT tasks: the goal is fixed, but its specific steps are hidden until you find
   and read something in the environment (e.g. "check the cookbook for the recipe", "find the
   clue and follow it"). Unlike reactive tasks, there is no ongoing stream of new instructions --
   just one lookup, then a normal fixed sequence. Do NOT guess the hidden specifics (e.g. which
   ingredients a recipe needs) before reading them. The plan's early steps should be to find and
   read/open the source of information; only after that can later steps be concrete. If the
   exact later steps can't be known yet, say so rather than inventing plausible-sounding ones.

For a FIXED-GOAL task, first identify every requirement the goal implies -- a state change (e.g.
hot/cold/clean), a quantity (e.g. "two" of something), or a specific supporting object -- then
produce a concise numbered plan that satisfies all of them. A state-change requirement is never
satisfied just by picking the object up near the right tool/appliance; it needs its own explicit
action step performed on the object.

Example (fixed-goal):
Task: Your task is to take the coin located in the canteen and put it into the box in the steam
room. A map is provided.
Requirements: find the coin in the canteen, navigate to the steam room, put the coin in the box
Plan:
1. go to the canteen
2. take the coin
3. navigate to the steam room (follow the map / room connections)
4. put the coin in the box

Example (reactive/instruction-following):
Task: Your task is to do exactly what Simon says.
Requirements: none fixed in advance -- the required action changes every turn based on the
environment's latest instruction
Plan:
1. Read the environment's current instruction carefully each turn.
2. Perform exactly the action it specifies right now, using the exact object/verb named.
3. Ignore any earlier guess about what the task might involve; always defer to the newest
   instruction, even if it contradicts a previous one.

Example (discover-then-act):
Task: Check the cookbook in the kitchen for the recipe, then cook and enjoy a meal.
Requirements: find and read the cookbook (specific ingredients/steps not yet known), then follow
the recipe exactly once read
Plan:
1. go to the cookbook and read it to learn the exact ingredients and preparation steps
2. gather each ingredient the recipe names (not yet known -- follow the recipe's own list)
3. prepare/cook following the recipe's exact instructions
4. eat the meal

Now do the same for the real task below. Output in exactly this format:
Requirements: <comma-separated list, or a note that requirements are revealed turn-by-turn or
after a lookup>
Plan:
1. <first step>
2. <second step>
...

If the task is reactive/instruction-following, the plan's first step should be the general
"read and follow the newest instruction" rule shown above, not a guess about the first
instruction's specific content. If the task is discover-then-act, do not invent specifics that
can only be known after reading the in-environment source. Output only the Requirements line and
the numbered plan, nothing else."""

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
    # max_tokens=1024: headroom for thinking-block content some models emit before the answer.
    return complete_with_usage(model, SYSTEM, prompt, max_tokens=1024)


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
    return complete_with_usage(model, REPLAN_SYSTEM, prompt, max_tokens=1024)
