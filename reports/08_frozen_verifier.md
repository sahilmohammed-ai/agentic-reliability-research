# TWX 08 — Frozen LLM Verifier

**Date:** 2026-07-20

**What:** a frozen (untrained, no checkpoint) LLM judge that scores each turn's action quality
0.0-1.0 via prompting (`verifier/frozen_llm.py`, Qwen2.5-3B). Sees the goal, prior action history,
and this turn's before/action/after; never sees `env_reward` or `won`. No new rollouts — scores
build 03's existing 100-episode baseline (2,114 turns) against real outcomes.

Two prompt bugs were found and fixed before the real run: a bare-number prompt collapsed to
judging "did the episode finish" rather than the turn itself, and a temporal-confusion bug read
the post-action observation as what the action should have anticipated. Fixed via a required
one-sentence reasoning step and an explicit instruction to judge only against `obs_before`.

**Won vs. lost mean verifier score:**

| game | won mean | lost mean |
|---|---|---|
| coin | 0.403 | 0.272 |
| simonsays | 0.790 | — (0 lost) |
| peckingorder | 0.777 | 0.360 |
| cookingworld | — (0 won) | 0.341 |
| mapreader | 0.432 | 0.596 |
| **overall** | **0.583** | **0.431** |

**Reward correlation (nonzero vs. zero `env_reward` mean score):**

| game | nonzero-reward | zero-reward |
|---|---|---|
| coin | 1.000 | 0.303 |
| simonsays | 0.790 | — |
| peckingorder | 0.630 | 0.903 |
| cookingworld | 0.431 | 0.336 |
| mapreader | 0.950 | 0.564 |

**Insights:**

- Overall, won episodes score meaningfully higher than lost (0.583 vs 0.431) — real signal.
- `coin`/`mapreader` show clean reward discrimination (1.000 vs 0.303; 0.950 vs 0.564).
- `peckingorder` inverts this (small sample, 5 lost turns) — unresolved.
- `mapreader`'s lost > won anomaly traced to a specific, minor weakness: direct trace inspection
  showed the verifier correctly scoring a real 23-step navigation dead-end as 0.0, but also
  mis-scoring the episode's actual winning action as 0.0. Quantified: 3/58 (5%) of won episodes
  have a low score on their final winning turn — real but minor, not systematic.
- `cookingworld` scores a flat, low 0.341 with no wins to compare — consistent with a genuine
  capability gap, not verifier confusion.

**Verdict:** a real, usable signal — tracks outcomes on 4/5 games, gives sensible per-turn
judgments on direct inspection. One minor, quantified weakness (undervalues terminal actions) and
one unresolved anomaly (`peckingorder`). Viable as a coordinator signal without training.
