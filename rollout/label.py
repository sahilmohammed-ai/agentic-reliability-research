"""
label rollout trajectories with per-turn q-values and advantages.

first-pass labeling: the only reward is the terminal win/loss signal, so the q-value for
each worker turn is the discounted return-to-go (gamma^(steps remaining) * terminal reward)
and the advantage is the change in value produced by the turn (value after minus value
before). once a trained verifier exists, these labels can be regenerated with proper td
bootstrapping the way agentprm does. for now this is pure monte carlo from the episode
outcome, no model needed.

terminal reward is derived from traj["won"] (1.0 if won, 0.0 if lost), NOT from the last
worker turn's raw env_reward. this matters across environments: alfworld's env_reward is 0
everywhere except a clean 1.0 on the winning step, but textworldexpress gives dense,
cumulative rewards (e.g. +0.25 per correct sub-step, negative on wrong ones), so its last
step's raw value can be anything (0.2, 0.25, -1.8, ...) depending on when the episode ended,
not a clean terminal signal. using won/lost directly keeps q-value scale identical across
every environment by construction, instead of silently training on inconsistent label scales
per environment (caught by inspecting build_10_textworldexpress: only 120 of 377 wins had
terminal env_reward == 1.0, the rest were 0.2/0.25; some losses had terminal env_reward as
extreme as -1.8).

repetition penalty (verifier v2, added after a research + independent-review pass): the MC
decay formula above has zero dependency on actual state/action, only on turn position and
episode outcome. confirmed via direct audit: a real trajectory repeating "examine countertop 1"
4x in a row (no progress) still got a SMOOTHLY INCREASING q_value across those turns, purely
from decay math. an initial fix idea (staged TD bootstrapping, train an initial model on these
same MC labels, freeze it, use it to relabel) was reviewed and REJECTED: the frozen model would
be trained on the same repetition-blind labels, so it can't supply the corrective signal needed,
it would just launder the same blind spot through an extra regression stage. the fix adopted
instead is a targeted, interpretable penalty:
  - detect a turn as part of a stall if the action repeats the previous action, or obs_before
    equals obs_after (the action produced literally no new information/state change).
  - only penalize from the 3rd consecutive stalled turn onward (turns 1-2 of a repeat are NOT
    penalized). checked directly against real won-episode data: e.g. "examine shelf 1" repeated
    4x in a row occurs in an episode that still won (re-examining once or twice is often a
    legitimate, harmless move, not a sign of being stuck); only sustained repetition (3+) is a
    reliable stall signal.
  - applied to ADVANTAGE only, never q_value. q_value is meant to stay a scale-consistent,
    roughly probability-of-success quantity; multiplying it by an ad hoc penalty would silently
    shift that distribution. advantage already encodes "how much did this turn change
    things," so a stall penalty belongs there.
  - explicitly scoped: this fixes literal/near-literal repetition, not general "unproductive but
    varied" behavior (e.g. cycling through different useless actions). documented as such, not
    presented as a comprehensive labeling fix.

hindsight sub-goal relabeling (verifier v2, second fix): even with the repetition penalty above,
every turn in a LOST episode still got q_value=0.0 flat (terminal_reward=0 collapses the whole
decay formula to zero regardless of turn position), and every turn in a WON episode still got a
purely positional decay curve with no dependency on whether the action taken was actually good.
researched real precedent (AgentPRM's multi-rollout MC re-estimates Q per state via many
re-rolled continuations; AgentHER/hindsight relabeling reinterprets what a trajectory actually
achieved). multi-rollout MC was REJECTED as impractical for this project: it needs ~16 additional
LLM-driven rollouts per VISITED STATE (not per episode), a 100-600x compute multiplier on top of
the 500 episodes already collected, and alfworld's wrapper (envs/alfworld_env.py) doesn't expose
mid-trajectory state branching, so implementing it would mean replaying every prior action to
reach a branch point first, an even worse cost. hindsight relabeling was adopted instead: it
requires zero new environment rollouts, just reprocessing the trajectories already on disk.
  - classifies each episode into one of 6 real task templates (confirmed via direct audit of
    task_goal strings: pick+place, clean+place, cool+place, heat+place, two-object, examine-
    with-desklamp -- these 6 cover all 500 build_10_alfworld episodes) and looks for the LAST
    turn where a genuine sub-goal action actually succeeded (pickup, placement, state-change, or
    desklamp-use, detected from real, consistent alfworld obs_after phrasing: "You pick up...",
    "You move/put...", "You heat/cool/clean... using...", "You turn on the desklamp...").
  - if a lost episode achieved a sub-goal at turn k, turns 1..k are labeled as if that partial
    trajectory had WON (terminal_reward=1.0, same decay formula, decayed back from turn k instead
    of from turn n), and turns after k fall back to the existing flat-zero treatment (nothing was
    achieved after that point, so there's nothing to credit).
  - explicitly, deliberately NOT applied to every loss: direct data audit found only 62.3% of
    losing episodes ever achieve so much as a single pickup (248/398); the remaining 37.7% never
    interact with any object at all (pure navigation/examination thrashing, e.g. train_0002.json,
    50 turns, never finds a mug). those episodes get no sub-goal to relabel toward, and keep the
    current flat-zero labels -- an honest reflection of "nothing happened here," not a bug to
    paper over with a forced heuristic.

usage:
    python -m rollout.label --in data/rollouts/build_5 --out data/labeled/build_5
"""

import argparse
import glob
import json
import os

# discount factor: how much each extra step to the goal reduces present value
GAMMA = 0.99

# repetition penalty: only fires from the 3rd consecutive stalled turn onward (see module
# docstring). STALL_MIN_RUN=3 means turns 1-2 of a repeat/no-op run are never penalized.
STALL_MIN_RUN = 3
STALL_PENALTY = 0.5  # advantage is scaled by this factor once a turn is in a penalized stall


def _is_stalled(turn: dict, prev_action: str | None) -> bool:
    """true if this turn produced no new information: the action repeats the previous action,
    or the observation didn't change at all."""
    same_action = prev_action is not None and turn["action"] == prev_action
    no_state_change = turn.get("obs_before") == turn.get("obs_after")
    return same_action or no_state_change


def _sub_goal_milestone(turn: dict) -> tuple[str, str] | None:
    """if this turn's action is a genuine alfworld sub-goal success (pickup, placement,
    state-change, or turning on a desklamp), return (milestone_type, object) e.g.
    ("take", "statue 1") or ("move", "statue 1"), else None. detected from the environment's own
    success-confirmation phrasing in obs_after (consistent across alfworld, confirmed via direct
    audit), not from the action string alone, so a FAILED take/put/heat/etc. (which alfworld
    narrates differently, e.g. "nothing happens") is not mistaken for a success.

    milestone_type is included alongside the object (not just the object alone) because pickup and
    placement are two DIFFERENT milestones for the same object and both deserve credit the first
    time each happens -- an earlier version keyed only on object, which wrongly skipped a real
    placement because a pickup of the same object had already been seen. dedup happens per
    (milestone_type, object) pair in _last_new_subgoal_turn, so a worker that places an object,
    then picks it back up and re-places the SAME object in a loop, gets credit for the first take
    and the first move, but not the second/third/fourth take-move cycle on the same object --
    confirmed as a real bug via direct data audit: train_0003.json, "put two statue in dresser",
    places statue 1 correctly at turn 5 then re-picks-up/re-places the SAME statue 4 more times,
    never placing a second one; picking it back up is churn, not further progress."""
    action = turn["action"]
    action_lower = action.lower()
    obs_after = turn.get("obs_after", "").lower()
    verb = action_lower.split()[0] if action_lower else ""

    succeeded = False
    if verb == "take":
        succeeded = obs_after.startswith("you pick up")
    elif verb in ("put", "move"):
        succeeded = obs_after.startswith("you move") or obs_after.startswith("you put")
    elif verb in ("heat", "cool", "clean"):
        succeeded = obs_after.startswith(f"you {verb}")
    elif verb == "use" and "desklamp" in action_lower:
        succeeded = "turn on the desklamp" in obs_after
    if not succeeded:
        return None

    if verb == "use":
        return ("use", "desklamp")
    # action shape is "<verb> <object...> from/to/with <receptacle...>" or "<verb> <object...>"
    # (e.g. "take statue 1 from dresser 1", "move statue 1 to dresser 1", "heat mug 1 with
    # microwave 1"); the object is everything between the verb and the first from/to/with.
    words = action.split()[1:]
    obj_words = []
    for w in words:
        if w.lower() in ("from", "to", "with", "in", "on"):
            break
        obj_words.append(w)
    obj = " ".join(obj_words).lower() if obj_words else None
    return (verb, obj) if obj else None


def _last_new_subgoal_turn(worker_turns: list[dict]) -> int | None:
    """index (0-based) of the turn where the LAST NEWLY-ACHIEVED (milestone_type, object) pair was
    reached, or None if the episode never achieved one at all (see module docstring: ~37.7% of
    losses, pure thrashing with no object interaction -- those get no hindsight relabeling, not a
    forced heuristic). credits only the FIRST success per (milestone_type, object) pair (a
    repeat-loop re-achieving the SAME milestone on the SAME object again does not extend the credit
    window), but DOES extend the window forward for a genuinely NEW milestone (a different object,
    or a different milestone type on an object already seen, e.g. take then move) since that's real
    additional progress."""
    seen: set[tuple[str, str]] = set()
    last_new = None
    for i, turn in enumerate(worker_turns):
        milestone = _sub_goal_milestone(turn)
        if milestone is not None and milestone not in seen:
            seen.add(milestone)
            last_new = i
    return last_new


def label_trajectory(traj: dict, gamma: float = GAMMA) -> dict:
    """add q_value and advantage to every worker turn in one trajectory dict."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    n = len(worker_turns)
    won = bool(traj.get("won"))
    # terminal reward from the won/lost flag, not the last turn's raw env_reward (see
    # module docstring for why: raw env_reward isn't a consistent terminal signal across
    # environments with dense/cumulative rewards like textworldexpress).
    terminal_reward = 1.0 if won else 0.0

    # hindsight relabeling (see module docstring): for a LOST episode, find the last turn that
    # achieved a real sub-goal. turns up through that point are decayed as if THAT was the
    # episode's terminal win, instead of collapsing to flat zero. turns after it (if any) get no
    # credit, since nothing more was achieved. only applies to losses; a win already has a real
    # terminal_reward=1.0 and its own full-length decay curve.
    subgoal_end = None if won else _last_new_subgoal_turn(worker_turns)

    prev_value = 0.0
    prev_action = None
    stall_run = 0  # consecutive stalled turns seen so far, see _is_stalled()
    for i, turn in enumerate(worker_turns):
        if won:
            q = (gamma ** (n - 1 - i)) * terminal_reward
        elif subgoal_end is not None and i <= subgoal_end:
            # decay back from the achieved sub-goal turn, not from the true (failed) ending
            q = gamma ** (subgoal_end - i)
        else:
            q = 0.0
        turn["q_value"] = round(q, 6)

        advantage = q - prev_value
        if _is_stalled(turn, prev_action):
            stall_run += 1
            if stall_run >= STALL_MIN_RUN:
                advantage *= STALL_PENALTY
        else:
            stall_run = 0
        turn["advantage"] = round(advantage, 6)

        prev_value = q
        prev_action = turn["action"]

    return traj


def label_directory(in_dir: str, out_dir: str, gamma: float = GAMMA) -> None:
    """label every trajectory json in in_dir and write results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise SystemExit(f"no trajectory files found in {in_dir}")

    won = 0
    turns_labeled = 0
    for path in files:
        with open(path) as f:
            traj = json.load(f)

        traj = label_trajectory(traj, gamma)
        won += int(traj["won"])
        turns_labeled += sum(1 for t in traj["turns"] if t["role"] == "worker")

        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(traj, f, indent=2)

    print(f"labeled {len(files)} trajectories ({won} won) -> {out_dir}/")
    print(f"{turns_labeled} worker turns now carry q_value and advantage")


if __name__ == "__main__":
    # parse cli arguments and label rollouts
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, required=True, help="directory of rollout jsons")
    parser.add_argument("--out", dest="out_dir", type=str, required=True, help="directory for labeled jsons")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir, args.gamma)
