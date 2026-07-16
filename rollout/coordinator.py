"""
verifier-driven coordinator (coordinator v0): a staged, threshold-based action toolkit that
reacts to the live verifier's q-value signal. this is the mechanism validation step before any
ppo training: five actions (continue/retry/replan/backtrack/escalate) of increasing cost, picked
by a simple rule rather than a learned policy, so we can confirm each action path actually does
something sensible before training a policy to choose between them.

design: score the PREVIOUS turn's outcome (its obs_after / resulting state) at the start of each
new step, using the live verifier. track how many consecutive turns have scored below Q_THRESHOLD.
escalating streak length triggers escalating intervention:
  streak 0        -> continue (worker acts normally)
  streak == 1     -> retry    (re-invoke the worker this turn with a "reconsider" nudge, no thinker call)
  streak == 2     -> replan   (thinker.replan(), same mechanism as replan_mode="stagnation")
  streak in [3,4] -> backtrack (clear action-history context, thinker.plan() fresh, not replan)
  streak >= 5     -> escalate (swap worker model for this one turn only, then revert)

streak resets to 0 the turn a score comes back >= Q_THRESHOLD.

known gap, fixed here: the verifier scores each turn in isolation, with no memory of history, so
it cannot tell that an action has been repeated many times. a smoke test showed a worker settle
into repeating "examine desk 1" for 38+ straight turns at a STABLE q=0.192 (above threshold every
single time), so the streak never built and the coordinator sat on "continue" the whole time. fix:
also trigger on literal action repetition (the same check rollout/runner.py's mask/stagnation
modes already use), independent of the q-value streak. either condition alone is enough to
trigger escalation, so the coordinator catches both "the verifier thinks this is failing" and
"the worker is visibly stuck" even when the verifier's per-turn score does not reflect the latter.

FOUND BROKEN in build_11 (50-episode alfworld run, 9/50 won vs build_10's 20.4% baseline), FIXED
2026-07-16 (see reports/build_11.md for the full trace-level diagnosis, reports/build_12.md for
the re-test): the streak had no decay and no ceiling. normal mid-episode turns routinely score
0.01-0.09 (the offline calibration this threshold was picked from was dominated by short episodes
scored near their final winning action, not representative of a typical middle turn), so streak>=5
was reached by turn 6-9 in nearly every real episode, and a single low-but-typical turn thereafter
was enough to prevent the streak from ever fully unwinding back to "continue" (needed 5+
CONSECUTIVE healthy turns). in practice escalate became the effectively permanent state for the
rest of any episode past ~10 steps (confirmed: 39.7% of all build_11 worker turns were "escalate",
identical rate in won vs lost episodes). worse, backtrack's `action_history = []` side effect
(rollout/runner.py) discarded the worker's memory of what it already tried, visibly inducing
thrashing (open/close/examine loops) that the verifier then correctly scored as low, re-triggering
the same streak that caused the thrashing in the first place.

fix: (1) low_q_streak now DECREMENTS on a healthy turn instead of resetting to 0, so recovery is
gradual rather than needing a perfect run of 5 consecutive good turns; (2) backtrack no longer
clears action_history, so a fresh plan can route around already-tried actions instead of the
worker blindly repeating them. Q_THRESHOLD (0.10) and the five-tier action ladder itself are
unchanged; whether turn-level q-value can reliably drive this kind of detection at all remains an
open question (checked this session: neither a fixed threshold nor a trailing-average trend signal
cleanly separated real loops from normal turns on build_11's data, precision 30-42%, recall
23-40% either way) -- out of scope for this fix, revisit if build_12 still underperforms.
"""

from collections import Counter

# calibrated from real data (data/labeled/build_10_alfworld, 60-episode sample, scored with
# verifier_v2): won-episode turns have p10=0.743, median=0.976; lost-episode turns have
# median=0.030, p75=0.056. 0.10 sits clearly above the lost-episode range and clearly below
# the won-episode range. the original 0.15 guess (picked from a single hand-inspected example,
# not real percentiles) was too close to the lost distribution's upper tail and let most
# actually-failing turns read as "fine".
Q_THRESHOLD = 0.10  # below this, the trajectory is considered "unhealthy" for this turn

# same repetition check as rollout/runner.py's is_stagnating(): a short window with too few
# distinct actions means the worker is stuck cycling, regardless of what the verifier scores.
STAGNATION_WINDOW = 6
STAGNATION_MAX_UNIQUE = 3

ESCALATION_MODEL = "claude-sonnet-4-5-20250929"  # model swapped in for the escalate action


def _is_repeating(action_history: list[str]) -> bool:
    """true if the worker's recent actions show a short cycle repeating (literal repetition,
    not a verifier judgment). same threshold as rollout/runner.py's is_stagnating()."""
    if len(action_history) < STAGNATION_WINDOW:
        return False
    recent = action_history[-STAGNATION_WINDOW:]
    return len(set(recent)) <= STAGNATION_MAX_UNIQUE


class VerifierCoordinator:
    """tracks the low-q streak across an episode and decides the coordinator action per turn."""

    def __init__(self, verifier):
        self.verifier = verifier
        self.low_q_streak = 0
        self.last_q_value: float | None = None
        self.last_advantage: float | None = None

    def observe(
        self, task_goal: str, plan: str, obs_before: str, action: str, action_history: list[str]
    ) -> tuple[float, float]:
        """score a completed turn and update the streak. call this once per turn, right after
        the worker has acted, so the NEXT turn's action decision is informed by this result.

        action_history is the full action history including the action just taken, used only
        for the literal-repetition check (the verifier score itself doesn't see it)."""
        q_value, advantage = self.verifier.score(task_goal, plan, obs_before, action)
        self.last_q_value = q_value
        self.last_advantage = advantage

        # either signal alone bumps the streak: a low verifier score, or literal repetition
        # the verifier's per-turn score can't see (see module docstring). a healthy turn
        # DECREMENTS rather than resets to 0: normal mid-episode turns routinely score below
        # Q_THRESHOLD even in winning episodes, so a hard reset made recovery need 5+ perfect
        # consecutive turns, which almost never happens (see FIXED note below).
        if q_value < Q_THRESHOLD or _is_repeating(action_history):
            self.low_q_streak += 1
        else:
            self.low_q_streak = max(0, self.low_q_streak - 1)

        return q_value, advantage

    def decide_action(self) -> str:
        """returns one of: continue, retry, replan, backtrack, escalate."""
        streak = self.low_q_streak
        if streak == 0:
            return "continue"
        elif streak == 1:
            return "retry"
        elif streak == 2:
            return "replan"
        elif streak in (3, 4):
            return "backtrack"
        else:
            return "escalate"
