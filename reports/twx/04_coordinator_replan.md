# TWX 04 — Coordinator Replan (Fixed Coordinator v1)

**Date:** 2026-07-19

**Change:** first coordinator action, not learned yet — a fixed heuristic. `rollout/runner.py`'s
`run_coordinated_episode` replans (via the previously-unused `thinker.replan()`) when the worker
is stuck in a literal 3x-repeat (`_is_repeating`) or a short 2-4 period cycle (`_is_cycling`).
Both are pure string-pattern checks on `action_history`, no reward or verifier signal involved.

**Metrics (Qwen2.5-3B, coordinated loop):**

| game | won | win rate | avg steps | replan rate | avg replans/ep |
|---|---|---|---|---|---|
| coin | 11/20 | 55% | 27.1 | 45% (9/20) | 0.75 |
| simonsays | 20/20 | 100% | 5.0 | 0% (0/20) | 0.00 |
| peckingorder | 18/20 | 90% | 8.0 | 0% (0/20) | 0.00 |
| cookingworld | 0/20 | 0% | 28.3 | 20% (4/20) | 0.20 |
| mapreader | 9/20 | 45% | 35.6 | 70% (14/20) | 3.85 |

**Comparison to build 03 (same model, no coordinator):**

| game | build 03 | build 04 (coordinated) |
|---|---|---|
| coin | 65% | 55% |
| simonsays | 100% | 100% |
| peckingorder | 90% | 90% |
| cookingworld | 0% | 0% |
| mapreader | 35% | 45% |

**Insights:**

- `simonsays`/`peckingorder` never triggered a replan (0% rate) — both are short/fast games where
  the coordinator has no chance to fire, so it's a true no-op there, not a wash.
- `cookingworld` triggers real replans (20% of episodes) but stays at 0% won — the detectors catch
  literal/cyclic loops, but this game's actual failure mode is semantically-stuck-but-lexically
  -varied churn (confirmed via direct trace inspection pre-build), which these detectors can't
  see. Replanning around a loop that isn't the real problem doesn't help.
- `coin` regressed (65% → 55%) with a meaningful trigger rate (45% of episodes) — plausibly the
  coordinator interrupting otherwise-progressing plans, though n=20 leaves real uncertainty.
- `mapreader` improved (35% → 45%) with by far the highest trigger rate (70%, 3.85 replans/ep) —
  the one clear positive signal, on the one game where the coordinator actually got to do a lot of
  work.

**Verdict:** mixed, not a clear win. One real gain (`mapreader`), one real loss (`coin`), two
no-ops (too short to trigger), and no improvement on the hardest game (`cookingworld`, wrong
failure mode for this detector). A fixed, string-pattern coordinator has limited ceiling — it
can't tell a costly interruption from a needed one. This motivates moving to a signal-aware
(reward- or verifier-driven) trigger rather than iterating further on heuristic detectors.
