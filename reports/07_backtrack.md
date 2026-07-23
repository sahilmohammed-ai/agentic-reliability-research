# TWX 07 — Backtrack (Fixed Coordinator v4)

**Date:** 2026-07-20

**Change:** a third coordinator action. Same reward-stall trigger as build 06 (15 consecutive
non-positive-reward steps), but instead of masking or replanning, clears `action_history`
entirely — no memory of the stalled context, no escalation (`rollout/runner.py`'s
`run_backtrack_episode`).

**Metrics (Qwen2.5-3B, backtrack loop):**

| game | won | win rate | avg steps | backtrack rate | avg backtracks/ep |
|---|---|---|---|---|---|
| coin | 11/20 | 55% | 28.0 | 60% | 1.50 |
| simonsays | 20/20 | 100% | 5.0 | 0% | 0.00 |
| peckingorder | 18/20 | 90% | 8.0 | 0% | 0.00 |
| cookingworld | 0/20 | 0% | 28.9 | 50% | 1.15 |
| mapreader | 7/20 | 35% | 36.8 | 75% | 1.85 |

**5-way comparison (no-coordinator / replan / mask / reward-aware / backtrack):**

| game | build 03 | build 04 | build 05 | build 06 | build 07 |
|---|---|---|---|---|---|
| coin | 65% | 55% | 65% | 60% | 55% |
| simonsays | 100% | 100% | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% | 40% | 35% |

**Insights:**

- `cookingworld` stays at 0% under a third independent intervention type despite a real trigger
  rate (50%) — mask, reward-aware, and backtrack all fail identically, strong evidence the failure
  is a capability gap, not fixable by coordination.
- `coin` under backtrack (55%) matches replan's cost exactly, well below masking's 65% — clearing
  all context is about as disruptive as discarding the plan.
- `mapreader` (35%, matching the no-coordinator baseline) trails replan's gain (45%) despite the
  highest trigger rate (75%) — clearing context doesn't help re-navigate here.

**Verdict — closing the fixed-coordinator series (builds 04-07):** four trigger/intervention
combinations tested, no dominant strategy. Masking is consistently gentlest and safest; replan and
backtrack both carry real disruption costs on some games. `cookingworld` never moves under any of
them — a model capability gap, not a coordination problem. This is the strongest case yet for a
**learned** coordinator that can pick the right intervention per-situation, rather than one fixed
rule applied everywhere.
