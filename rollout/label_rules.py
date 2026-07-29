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
estimator, not a pure environment-outcome value estimator), stated plainly, not hidden. this
DOES give real move-vs-move contrast that the old labels couldn't: two different actions at the
same state can now get different rule scores (e.g. one repeats a known-failed action, the other
doesn't), which the env_reward-based labels could never provide because both would score identical
(usually 0.0) reward regardless of which was chosen.

rule categories, confirmed against real data (data/labeled/v5_train_combined/):
  FAILED/NO-OP action: obs_after matches a known failure phrase ("that is already open/closed",
    "you can't move there", "the door is closed", or obs_after == obs_before, i.e. nothing changed)
    -> negative score. confirmed common: "That is already open." alone is 628/first-100-episodes'
    worth of short obs_after strings, "You can't move there, the door is closed." is 119.
  REPEATED FAILURE: the exact same action was already taken earlier in this episode AND failed
    both times -> larger negative score (worse than a first-time failure; the agent should have
    learned from the first failure).
  GOAL PROGRESS: obs_after or action mentions a token from task_goal (the target object/subject
    the task asks for), or action is "take X"/"read X"/similar where X appears in the goal ->
    positive score. confirmed pattern: "You take the coin." on the winning action of a real won
    coin episode, "You take the TV remote control." (goal-irrelevant, no credit) on a real lost one.
  everything else (a normal, non-failing, non-goal action -- most exploration/movement) -> 0,
    neutral, matching the old labels' typical value for a normal turn.

usage:
    python -m rollout.label_rules --in data/rollouts/v6_train/coin --out data/labeled/v6_rules/coin
"""

import argparse
import glob
import json
import os
import re

GAMMA = 0.99  # discount factor, same as label_twx.py, for consistency of shape

# confirmed common failure/no-op phrases across real data (see module docstring for counts).
# lowercase substring match against obs_after.
_FAILURE_PHRASES = (
    "that is already open",
    "that is already closed",
    "can't move there",
    "you can't go that way",
    "nothing happens",
    "i don't understand",
    "can't see any",
)

_GOAL_ACTION_VERBS = ("take", "read", "open", "unlock")


def _is_failed_or_noop(obs_before: str, obs_after: str) -> bool:
    """a turn is a failed/no-op action if the observation text explicitly signals failure, or if
    the world state didn't visibly change at all (obs_after identical to obs_before -- a stronger,
    more general catch-all than enumerating every possible failure phrase, since TextWorldExpress
    games phrase failures differently across games; this catches any of them uniformly)."""
    oa_lower = obs_after.strip().lower()
    if any(phrase in oa_lower for phrase in _FAILURE_PHRASES):
        return True
    return obs_after.strip() == obs_before.strip()


_GOAL_VERBS = ("take", "put", "unlock", "open", "read", "slice", "eat", "touch", "examine", "smell")


def _mentions_goal(task_goal: str, action: str, obs_after: str) -> bool:
    """does the ACTION ITSELF (a verb + object, e.g. "take coin") both (a) start with a real
    goal-completion verb and (b) name a content word that also appears in the task goal?

    NOT a loose "does any goal word appear anywhere in the surrounding text" check -- an earlier
    version of this rule did that and was confirmed WRONG on real data: on mapreader, an agent
    bouncing uselessly between "bathroom" and "steam room" got +1.00 on every turn because "room"/
    "steam" happened to appear in both the goal text and the room-name narration; on simonsays,
    "Simon says, 'X'" appears in nearly every obs_after because it's the game's turn-announcement
    format, so "simon" (from the goal "do exactly what Simon says") matched on ~97% of ALL turns
    regardless of whether the action taken was actually correct. Restricting the match to the
    ACTION STRING itself, and requiring it start with a genuine goal-completion verb, avoids both:
    incidental scenery/narration words in obs_after can no longer trigger a false positive, and a
    plain "move"/"open door" (this project's most common actions) can never match regardless of
    what appears in the resulting text."""
    action_lower = action.lower().strip()
    if not any(action_lower.startswith(v) for v in _GOAL_VERBS):
        return False

    stopwords = {
        "the", "a", "an", "and", "or", "to", "your", "task", "is", "you",
        "it", "find", "once", "then", "take", "in", "on", "of", "with",
        "exactly", "what", "says", "do",
    }
    goal_words = {w for w in re.findall(r"[a-z]+", task_goal.lower()) if w not in stopwords and len(w) > 2}
    if not goal_words:
        return False
    action_words = set(re.findall(r"[a-z]+", action_lower))
    return bool(goal_words & action_words)


_SIMON_SAYS_RE = re.compile(r"simon says,?\s*'([^']+)'", re.IGNORECASE)


def _matches_simon_instruction(obs_before: str, action: str) -> bool | None:
    """simonsays-specific rule: obs_before literally states the required action ("Simon says,
    'take grapefruit'."), so correctness is checkable EXACTLY, not via word-overlap -- a much
    stronger signal than the generic goal-word rule, which is meaningless here since the game's
    static task_goal ("do exactly what Simon says") names no object at all, only a per-turn
    instruction embedded in the observation text. returns None if obs_before doesn't contain a
    Simon-says instruction (so the generic rule can still apply as a fallback, e.g. this game's
    OWN failure-phrase turns), True/False for an instruction-following match/mismatch."""
    m = _SIMON_SAYS_RE.search(obs_before)
    if not m:
        return None
    return m.group(1).strip().lower() == action.strip().lower()


def _score_turn(
    task_goal: str, obs_before: str, action: str, obs_after: str, prior_failed_actions: set[str],
) -> float:
    """rule score for one turn. returns a float in roughly [-1.0, 1.0]:
      +1.0: action clearly advances the stated goal (take/read/open the target object), OR
            (simonsays only) exactly matches the instruction just given
      -0.6: action repeats a PRIOR failed action verbatim (should have learned from the first try)
      -0.5: (simonsays only) action does NOT match the instruction just given -- a clean, exact
            mismatch, checkable with certainty since the correct answer is stated verbatim
      -0.3: action failed or was a no-op (first occurrence, not yet a repeat)
       0.0: a normal, non-failing, non-goal-relevant action (most exploration/movement)
    goal-progress is checked first: an action that both matches a failure phrase AND happens to
    mention a goal word (rare, but e.g. "that is already open" on a door that happens to share a
    word with the goal) should not get positive credit -- so failure/repeat checks take priority.
    the simonsays instruction check takes priority over the generic goal-word rule when it applies
    (returns non-None), since it's an exact, certain signal rather than a heuristic word match."""
    simon_match = _matches_simon_instruction(obs_before, action)
    if simon_match is True:
        return 1.0
    if simon_match is False:
        return -0.5

    failed = _is_failed_or_noop(obs_before, obs_after)
    if failed:
        if action in prior_failed_actions:
            return -0.6
        return -0.3
    if _mentions_goal(task_goal, action, obs_after):
        return 1.0
    return 0.0


def _mc_returns_to_go(scores: list[float], gamma: float) -> list[float]:
    """same discounted-sum-backward shape as label_twx.py's MC return-to-go, applied to RULE
    scores instead of env_reward -- keeps the verifier's training TARGET SHAPE (a return-to-go,
    not a per-turn score) unchanged, so verifier/model.py and verifier/train.py need no changes,
    only the label source changes."""
    returns = [0.0] * len(scores)
    running = 0.0
    for t in range(len(scores) - 1, -1, -1):
        running = scores[t] + gamma * running
        returns[t] = running
    return returns


def label_trajectory(traj: dict, gamma: float = GAMMA) -> dict:
    """add rule-based q_value and advantage to every worker turn, replacing env_reward-derived
    labels. same output shape as label_twx.py (q_value = return-to-go, advantage = one-step
    delta), different label SOURCE."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    task_goal = traj["task_goal"]

    scores = []
    prior_failed_actions: set[str] = set()
    for turn in worker_turns:
        score = _score_turn(task_goal, turn["obs_before"], turn["action"], turn["obs_after"], prior_failed_actions)
        scores.append(score)
        if score < 0:
            prior_failed_actions.add(turn["action"])

    q_values = _mc_returns_to_go(scores, gamma)

    prev_value = 0.0
    for turn, q, rule_score in zip(worker_turns, q_values, scores):
        turn["q_value"] = round(q, 6)
        turn["advantage"] = round(q - prev_value, 6)
        turn["rule_score"] = round(rule_score, 6)  # the raw per-turn rule score, kept for
                                                     # diagnostics/inspection, not used by training
        prev_value = q

    return traj


def label_directory(in_dir: str, out_dir: str, gamma: float = GAMMA) -> None:
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

        traj = label_trajectory(traj, gamma)
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
    parser.add_argument("--gamma", type=float, default=GAMMA)
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir, args.gamma)
