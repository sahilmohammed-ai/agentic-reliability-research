# Build 11

**Date:** 2026-07-16

**Overview:** First multi-episode (n=50) test of coordinator v0 (`replan_mode=verifier`), on ALFWorld,
collected on Lightning AI. Result: 9/50 won (18.0%), at or below build_10's mask-mode baseline of
102/500 (20.4%). Root-caused below: not a verifier-quality problem, a coordinator design flaw that
turns into a self-reinforcing feedback loop.

**Environment:** ALFWorld (TextWorld backend), train split, 50 episodes.

**Agent setup:**
- Thinker + Worker: Qwen2.5-3B-Instruct, HF Transformers full precision.
- Coordinator: v0 (`rollout/coordinator.py`), live-verifier-driven, `checkpoints/verifier_v1`,
  `Q_THRESHOLD=0.10`.

**Metrics:**
- 50 episodes, 9 won (18.0%), 41 lost.
- Won episodes averaged 12.3 steps. Lost episodes: all 41/41 hit the 50-step cap.
- Coordinator action histogram across all 2,161 worker turns: continue 35.4%, retry 9.1%,
  replan 7.0%, backtrack 8.8%, **escalate 39.7%**.
- Escalate rate is nearly identical between won (47.75 per 100 turns) and lost (39.27 per 100
  turns) episodes, i.e. escalating does not distinguish outcome at all.

**Root cause (confirmed via direct verifier re-scoring, not inference from the histogram alone):**

The verifier itself is not the problem. Re-scored a range of build_10 training examples live
through `verifier/infer.py` (same checkpoint used in this run): a 30-turn win scored 0.887 (label
1.0), 11 of 12 two-object-task wins (e.g. "put two vase in coffeetable", "find two toiletpaper and
put them in toilet") scored 0.4-1.1 in line with their labels. The verifier generalizes correctly
to long episodes and to two-object tasks.

What's actually broken is `rollout/coordinator.py`'s `low_q_streak` design:
1. **Normal mid-episode Q-values sit around 0.01-0.09**, even in episodes that go on to win. The
   offline calibration (`verifier_v1.md`'s 60-episode sample: won-turn median 0.976) was skewed by
   short, clean trajectories scored mostly near their final winning action; it is not representative
   of a typical middle-of-episode turn. Confirmed directly: even the literal winning final action
   of a won build_11 episode (`train_0009`, "put two cd in drawer") scored q=0.010 live.
2. **The streak has no decay and no ceiling.** Reaching streak>=5 requires 5+ *consecutive* turns
   below 0.10; given (1), this happens by turn 6-9 in nearly every real episode. Once escalate is
   reached, a single normal (low-but-typical) turn is enough to prevent the streak from ever fully
   unwinding back to `continue`, since any turn below threshold restarts the countdown. In practice
   escalate becomes the coordinator's effectively permanent state for the rest of the episode: e.g.
   `train_0000` escalates turns 6-9, drops out for one turn (10), then escalates continuously from
   turn 15 onward for the rest of a 50-step episode.
3. **This creates a feedback loop that damages otherwise-fine trajectories.** `train_0009` (won,
   despite this): after the coordinator's retry/backtrack interventions kick in around turn 6, the
   worker enters a 12-turn open/close/examine/take/put thrashing loop on the same drawer (turns
   6-17) before organically moving on to the second object at turn 18 and winning. The verifier
   correctly scores this thrashing as low-Q; that low score is what feeds the streak; the streak is
   what triggered backtrack/escalate in the first place. The coordinator's own intervention is
   what produced the low-Q behavior it then reacts to, not an independent signal of pre-existing
   trouble.
4. **Every one of the 41 losses ran the full 50-step cap.** The escalate-lock state doesn't recover
   and doesn't fail fast either; it just burns the entire step budget.

**Notes:**
- This is a genuine negative result for the current threshold/streak mechanism, not evidence
  against the verifier or against the coordinator-v0 concept generally. The verifier's underlying
  signal is directionally sound; the policy consuming that signal (a monotonic streak counter with
  no decay, calibrated on an unrepresentative offline sample) is what needs to change before another
  collection run.
- Immediate fix candidates, not yet implemented: (a) recalibrate Q_THRESHOLD (or better, use a
  rolling/relative measure, e.g. comparing to a trailing average rather than a fixed cutoff) against
  a live, mid-episode Q-value distribution instead of the end-of-episode-heavy offline sample;
  (b) add streak decay (e.g. decrement on a good turn rather than a full reset-or-nothing), so a
  single normal dip doesn't erase recovery progress; (c) reconsider whether `backtrack` and
  `escalate`'s own side effects (clearing action history, forcing fresh plans) are what's inducing
  the thrashing behavior the verifier then penalizes, independent of the threshold question.
- This result and its root cause are the direct next input into fixing `rollout/coordinator.py`
  before any further multi-episode `replan_mode=verifier` collection.
