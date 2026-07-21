# TWX 05 — Masking (Fixed Coordinator v2)

**Date:** 2026-07-19

**Change:** an alternative fixed-coordinator intervention on the same trigger as build 04's
replanning. `rollout/runner.py`'s `run_masked_episode` uses the identical detectors
(`_is_repeating`, `_is_cycling`) but, instead of calling `thinker.replan()`, removes the looping
action(s) from `admissible_commands` (`_looping_actions`) so the worker physically cannot repeat
them and must pick something else — no new plan, just a narrower choice.

**Metrics (Qwen2.5-3B, masked loop):**

| game | won | win rate | avg steps | mask rate | avg masks/ep |
|---|---|---|---|---|---|
| coin | 13/20 | 65% | 27.8 | 45% (9/20) | 0.75 |
| simonsays | 20/20 | 100% | 5.0 | 0% (0/20) | 0.00 |
| peckingorder | 18/20 | 90% | 8.0 | 0% (0/20) | 0.00 |
| cookingworld | 0/20 | 0% | 28.4 | 20% (4/20) | 0.20 |
| mapreader | 5/20 | 25% | 38.9 | 70% (14/20) | 3.75 |

**3-way comparison (build 03 no-coordinator / build 04 replan / build 05 masking):**

| game | build 03 | build 04 (replan) | build 05 (masking) |
|---|---|---|---|
| coin | 65% | 55% | 65% |
| simonsays | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% |

**Insights:**

- `coin` fully recovers under masking (65% → 55% → 65%) — masking is the gentler intervention:
  removing one bad option preserves the rest of an otherwise-working plan, where a full replan
  discarded it.
- `simonsays`/`peckingorder` are identical across all three builds — trigger rates are ~0% at this
  episode length, so neither intervention gets a chance to matter.
- `cookingworld` stays at 0% under masking too, despite real triggers (20% of episodes, matching
  build 04's rate exactly). Confirms the bottleneck is the *detector* — it only catches literal
  and cyclic loops, not this game's actual semantic-churn failure mode — not which intervention
  responds to it.
- `mapreader` is the one place masking loses: 25%, below both build 03's baseline (35%) and build
  04's replan result (45%), despite the highest trigger rate of any game (70%, 3.75 masks/ep).
  Removing options without giving fresh strategy seems to leave the worker floundering among
  what's left, rather than helping — this game likely needs the re-navigation planning a replan
  provides, not just a narrower choice set.

**Verdict:** masking beats or matches replanning everywhere except `mapreader`, where it clearly
loses. Neither intervention is strictly dominant — masking is the safer default (never worse than
no-coordinator on 4/5 games, actively better on `coin`), but `mapreader` shows a real case where
discarding the plan and re-navigating is worth more than just narrowing the choice set. This
suggests the eventual learned coordinator should have both actions available and pick between them
per-situation, rather than one fixed intervention type applied everywhere.
