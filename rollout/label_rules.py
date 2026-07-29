"""
label TextWorldExpress rollout trajectories by a HAND-WRITTEN RULE SCORE instead of real
environment reward -- replaces rollout/label_twx.py's env_reward-based MC return for verifier_v6.

why: builds 13's coordinator saga (see .info/CLAUDE.md) traced the coordinator's plateau to a
confirmed, two-layer problem in the env_reward-based labels: (1) the advantage formula
(q_value_t - q_value_{t-1}) doesn't depend on the action taken at t; (2) more fundamentally,
env_reward itself is ~0 on almost every non-terminal turn in all 4 training games (confirmed via
direct data audit: coin has 0.00% nonzero non-terminal reward across 8,881 real turns; a LOST coin
episode's q_value is IDENTICALLY 0.0 at every single one of its turns, win or lose, no signal
anywhere). Three candidate relabeling fixes (outcome-based labeling, verifier self-bootstrapping,
same-state branching collection) were tried or reasoned through and confirmed NOT to fix this --
none of them can manufacture per-action information that isn't in env_reward to begin with.

this labeler abandons env_reward as the training target entirely and scores each turn by explicit,
checkable RULES over what's already recorded per turn (obs_before, action, obs_after,
admissible_commands) -- a real, honest change in what the verifier IS (a rule-conformance
estimator, not a pure environment-outcome value estimator), stated plainly, not hidden.

v2 (2026-07-28), after an independent review found real bugs in v1 by reading actual data, not
just the docstring's claims -- fixed here, not glossed over:
  - v1's generic goal-word-match rule gave +1.0 to WRONG-OBJECT actions on mapreader (putting junk
    in the box scored the same as putting the coin in the box, since both share "box" with the
    goal) and was essentially meaningless on peckingorder (the real per-turn target lives in the
    game's OWN "read instructions book" response text, not the static task_goal, so "read
    instructions book" itself scored +1.0 on 46.7% of ALL turns while correct vs incorrect "take"
    actions were BOTH scored 0.0, no discrimination at all). Fixed with GAME-SPECIFIC exact-target
    tracking for peckingorder (mirrors the simonsays pattern: track the CURRENT stated instruction,
    check the NEXT action against it exactly) and a tightened mapreader rule (only "coin"/"box" are
    real targets, not any shared goal word).
  - v1's repeated-failure penalty used raw action-string matching, so "open door to west" failing
    in one room falsely counted as a "repeat" of an unrelated "open door to west" failing in a
    DIFFERENT room later. Fixed by scoping prior_failed_actions per (room-identifying prefix of
    obs_before, action) instead of action alone.
  - v1 ran rule scores through the same MC-return-to-go discounting as the old env_reward labels,
    which an independent review found PARTIALLY reintroduces the original bug: discounting a rare
    +1.0 spike backward through many neutral turns makes q_value dominated by "how many steps until
    the nearest spike" (position), diluting the real same-state contrast the rule scores provide.
    Fixed by using the RAW per-turn rule score directly as q_value (no discounting) -- see
    label_trajectory()'s docstring for the reasoning.

usage:
    python -m rollout.label_rules --in data/rollouts/v5_train/coin --out data/labeled/v6_rules/coin
"""

import argparse
import glob
import json
import os
import re

# confirmed comprehensive against real data (independent review scanned all short obs_after
# strings across all 4 games and found no uncaught no-progress messages beyond these + the
# obs_after==obs_before catch-all).
_FAILURE_PHRASES = (
    "that is already open",
    "that is already closed",
    "can't move there",
    "you can't go that way",
    "nothing happens",
    "i don't understand",
    "can't see any",
)


def _is_failed_or_noop(obs_before: str, obs_after: str) -> bool:
    """a turn is a failed/no-op action if the observation text explicitly signals failure, or if
    the world state didn't visibly change at all (obs_after identical to obs_before -- a stronger,
    more general catch-all than enumerating every possible failure phrase, since TextWorldExpress
    games phrase failures differently across games; this catches any of them uniformly)."""
    oa_lower = obs_after.strip().lower()
    if any(phrase in oa_lower for phrase in _FAILURE_PHRASES):
        return True
    return obs_after.strip() == obs_before.strip()


def _room_context(obs_before: str) -> str:
    """a coarse room-identifying prefix of the observation, used to scope the repeated-failure
    check so "open door to west" failing in room A is never confused with an UNRELATED "open door
    to west" failing later in room B (v1 bug: exact action-string matching alone can't tell two
    different doors apart when their action strings happen to be identical). the first ~40 chars
    of obs_before reliably contains the room name/description opener in these games (e.g. "You are
    in the kitchen...", "You are in the bathroom...") -- coarse but sufficient to distinguish rooms
    without needing a real state-tracking parser."""
    return obs_before.strip()[:40].lower()


_GOAL_VERBS = ("take", "put", "unlock", "open", "read", "slice", "eat", "touch", "examine", "smell")


def _mentions_goal_generic(task_goal: str, action: str) -> bool:
    """GENERIC fallback rule (used for coin, and as a fallback elsewhere): does the ACTION ITSELF
    (not the surrounding observation text -- seeobs_after false-positives fixed in v1) start with a
    real goal-completion verb AND name a content word shared with the task goal? adequate for coin,
    whose goal only ever names one real target ("find the coin ... take it") with no other objects
    in play that could cause a false-positive collision. NOT used for mapreader/peckingorder, which
    need game-specific exact-target rules instead (see below) because their generic goal text
    contains words ("box", "instructions") that collide with irrelevant objects/actions."""
    action_lower = action.lower().strip()
    if not any(action_lower.startswith(v) for v in _GOAL_VERBS):
        return False
    stopwords = {
        "the", "a", "an", "and", "or", "to", "your", "task", "is", "you",
        "it", "find", "once", "then", "take", "in", "on", "of", "with",
    }
    goal_words = {w for w in re.findall(r"[a-z]+", task_goal.lower()) if w not in stopwords and len(w) > 2}
    if not goal_words:
        return False
    action_words = set(re.findall(r"[a-z]+", action_lower))
    return bool(goal_words & action_words)


_SIMON_SAYS_RE = re.compile(r"simon says,?\s*'([^']+)'", re.IGNORECASE)


def _matches_simon_instruction(obs_before: str, action: str) -> bool | None:
    """simonsays: obs_before literally states the required action ("Simon says, 'take
    grapefruit'."), checkable EXACTLY. returns None if obs_before has no Simon-says instruction
    (falls through to the generic/failure rules), True/False for a match/mismatch. confirmed
    correct against real data: 97.7% +1.0 / 2.3% -0.5 on a real sample, mismatches verified against
    actual `Incorrect!` game-over turns."""
    m = _SIMON_SAYS_RE.search(obs_before)
    if not m:
        return None
    return m.group(1).strip().lower() == action.strip().lower()


_PECKINGORDER_TASK_RE = re.compile(r"current task is to (\w+) the (\w+)", re.IGNORECASE)


def _matches_peckingorder_instruction(pending_target: tuple[str, str] | None, action: str) -> bool | None:
    """peckingorder: the real per-turn target is NOT in the static task_goal ("read and follow the
    instructions book... repeat this") -- it's revealed by "read instructions book"'s OWN response
    text ("The current task is to take the pineapple."), confirmed identical phrasing across every
    real sampled episode. v1's bug: it scored "read instructions book" itself as +1.0 (matching
    "instructions"/"book" against the goal) and gave BOTH correct and incorrect "take" actions a
    flat 0.0, no discrimination. Fixed: label_trajectory() tracks pending_target (verb, object)
    parsed from the most recent "read instructions book" response, and this function checks
    whether the CURRENT action matches it exactly -- same pattern as simonsays. returns None if no
    target is currently pending (falls through to generic/failure rules, e.g. covers "read
    instructions book" itself, which gets a small fixed credit in _score_turn since it's a
    necessary step, not the goal-completing one)."""
    if pending_target is None:
        return None
    verb, obj = pending_target
    action_lower = action.strip().lower()
    return action_lower == f"{verb} {obj}"


def _mentions_mapreader_target(task_goal: str, action: str) -> bool:
    """mapreader: the goal is always "take the coin ... put it into the box" -- v1's bug was
    crediting ANY take/put action that shared a word ("box") with the goal, so "put dirty brown
    skirt in box" scored the same as "put coin in box" (confirmed on real data: a LOST episode got
    six false +1.0 turns this way). Fixed: only "coin" (for take) and "box" as the ACTUAL
    destination of a put (not just present in the goal) count -- "coin" and "box" are the only two
    real targets in every mapreader instance, so this is a safe, non-heuristic tightening, not a
    game-specific special case that could break on a differently-phrased instance."""
    action_lower = action.strip().lower()
    if action_lower.startswith("take") and "coin" in action_lower:
        return True
    if action_lower.startswith("put") and "coin" in action_lower and "box" in action_lower:
        return True
    return False


_OSCILLATION_WINDOW = 4  # how many recent obs_after strings to check for a revisited state


def _is_oscillating(obs_after: str, recent_observations: list[str]) -> bool:
    """catches loops that _is_failed_or_noop can't: each individual move can genuinely SUCCEED
    (a real room transition, not a failure message) while the agent is still just bouncing between
    a small set of already-seen states with no new progress. confirmed as a real, common, currently
    -uncaught pattern on real data: a lost mapreader episode's final 25 turns are pure "move east" /
    "move west" alternation between two rooms, each individual move succeeding, all scored neutral
    0.0 by the failure/goal rules alone. checks whether obs_after exactly matches one already seen
    in the last _OSCILLATION_WINDOW turns -- a real, visible loop, not a heuristic guess."""
    return obs_after.strip() in recent_observations


def _score_turn(
    game: str, task_goal: str, obs_before: str, action: str, obs_after: str,
    prior_failed: set[tuple[str, str]], pending_peckingorder_target: tuple[str, str] | None,
    recent_observations: list[str],
) -> float:
    """rule score for one turn, roughly in [-1.0, 1.0]. game-specific exact-instruction checks
    (simonsays, peckingorder) take priority when they apply, since they're certain, not heuristic.
    failure/repeat checks come next, then oscillation, then goal-word matching (coin's generic
    rule, mapreader's tightened rule) as the fallback for the remaining games."""
    if game == "simonsays":
        m = _matches_simon_instruction(obs_before, action)
        if m is True:
            return 1.0
        if m is False:
            return -0.5

    if game == "peckingorder":
        action_lower = action.strip().lower()
        if action_lower == "read instructions book":
            return 0.2  # a necessary step, not the goal-completing one -- small fixed credit,
                         # not the +1.0 v1 wrongly gave it (that was masking real discrimination)
        if action_lower.startswith("take") and pending_peckingorder_target is None:
            # a "take" with NO pending instruction -- confirmed a real, distinct failure mode on
            # real data: a 3-turn LOST episode (read -> correct take -> ANOTHER take with no new
            # instruction in between) ends in a loss precisely because the agent acted without
            # reading a fresh instruction first. v2 scored this 0.0 (no signal at all, since the
            # instruction-match check has nothing to compare against) -- the actual mistake here
            # isn't "wrong object," it's "acted blind," and that's exactly as real a failure as a
            # wrong-object mismatch, so it gets the same penalty.
            return -0.5
        m = _matches_peckingorder_instruction(pending_peckingorder_target, action)
        if m is True:
            return 1.0
        if m is False:
            return -0.5

    failed = _is_failed_or_noop(obs_before, obs_after)
    if failed:
        key = (_room_context(obs_before), action)
        if key in prior_failed:
            return -0.6
        return -0.3

    if _is_oscillating(obs_after, recent_observations):
        return -0.3  # same tier as a first-time failure -- a real, visible lack of progress,
                      # even though the individual action technically succeeded

    if game == "mapreader":
        if _mentions_mapreader_target(task_goal, action):
            return 1.0
        return 0.0

    if _mentions_goal_generic(task_goal, action):
        return 1.0
    return 0.0


def label_trajectory(traj: dict) -> dict:
    """add rule-based q_value/advantage to every worker turn, replacing env_reward-derived labels.
    q_value is the RAW per-turn rule score directly (NOT a discounted MC return-to-go) -- an
    independent review found that discounting a sparse rule-score spike backward through many
    neutral turns (the same formula shape as label_twx.py) makes q_value dominated by proximity to
    the nearest spike (position), diluting the real same-state action contrast the rule scores
    provide, which is the entire point of this labeler. advantage is still a one-step delta
    (q_value_t - q_value_{t-1}) for shape-compatibility with verifier/model.py's 2-output head, but
    is now ALSO genuinely action-dependent at every turn, since q_value itself is."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    task_goal = traj["task_goal"]
    game = traj.get("turns", [{}])[0].get("metadata", {}).get("gameName", "") or _infer_game(traj)

    scores = []
    prior_failed: set[tuple[str, str]] = set()
    pending_peckingorder_target: tuple[str, str] | None = None
    recent_observations: list[str] = []
    for turn in worker_turns:
        score = _score_turn(
            game, task_goal, turn["obs_before"], turn["action"], turn["obs_after"],
            prior_failed, pending_peckingorder_target, recent_observations,
        )
        recent_observations.append(turn["obs_before"].strip())
        if len(recent_observations) > _OSCILLATION_WINDOW:
            recent_observations.pop(0)
        scores.append(score)

        if score < 0:
            prior_failed.add((_room_context(turn["obs_before"]), turn["action"]))

        if game == "peckingorder":
            action_lower = turn["action"].strip().lower()
            if action_lower == "read instructions book":
                m = _PECKINGORDER_TASK_RE.search(turn["obs_after"])
                if m:
                    pending_peckingorder_target = (m.group(1).lower(), m.group(2).lower())
            elif action_lower.startswith("take") and pending_peckingorder_target is not None:
                # a "take" attempt (whether it matched the target or not) resolves/consumes the
                # pending instruction -- the next instruction only arrives via another
                # "read instructions book" call. bug fixed here (v2->v3, caught via direct data
                # audit): the PREVIOUS version cleared pending_peckingorder_target on every
                # non-"read" turn unconditionally, including the SAME turn that had just correctly
                # consumed it -- which accidentally also wiped it one turn too early in the normal
                # case, but more importantly meant a WRONG "take X" (a real mismatch) still cleared
                # the target immediately, so the NEXT turn (if the agent tried again) had nothing
                # to check against. Now the target only clears on an actual "take" attempt, right
                # or wrong -- matching the real game mechanic where one "take" per instruction is
                # the norm and a fresh "read instructions book" is needed for the next target.
                pending_peckingorder_target = None

    prev_value = 0.0
    for turn, score in zip(worker_turns, scores):
        turn["q_value"] = round(score, 6)
        turn["advantage"] = round(score - prev_value, 6)
        turn["rule_score"] = round(score, 6)  # identical to q_value now; kept as an explicit,
                                                # clearly-named field for diagnostics/inspection
        prev_value = score

    return traj


def _infer_game(traj: dict) -> str:
    """fallback game-name detection if a turn's metadata doesn't carry gameName (e.g. rollouts
    collected before that field existed) -- infer from characteristic task_goal phrasing."""
    goal = traj.get("task_goal", "").lower()
    if "simon" in goal:
        return "simonsays"
    if "instructions book" in goal:
        return "peckingorder"
    if "put it into the box" in goal:
        return "mapreader"
    return "coin"


def label_directory(in_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise SystemExit(f"no trajectory files found in {in_dir}")

    won = 0
    turns_labeled = 0
    score_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for path in files:
        with open(path) as f:
            traj = json.load(f)

        traj = label_trajectory(traj)
        won += int(traj["won"])
        worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
        turns_labeled += len(worker_turns)
        for t in worker_turns:
            if t["rule_score"] > 0:
                score_counts["positive"] += 1
            elif t["rule_score"] < 0:
                score_counts["negative"] += 1
            else:
                score_counts["neutral"] += 1

        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(traj, f, indent=2)

    print(f"labeled {len(files)} trajectories ({won} won) -> {out_dir}/")
    print(f"{turns_labeled} worker turns: "
          f"{score_counts['positive']} positive ({score_counts['positive']/turns_labeled*100:.1f}%), "
          f"{score_counts['negative']} negative ({score_counts['negative']/turns_labeled*100:.1f}%), "
          f"{score_counts['neutral']} neutral ({score_counts['neutral']/turns_labeled*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, required=True)
    parser.add_argument("--out", dest="out_dir", type=str, required=True)
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir)
