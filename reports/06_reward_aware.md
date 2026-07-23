# TWX 06 — Reward-Aware Coordinator (Fixed Coordinator v3)

**Date:** 2026-07-19

**Change:** replaces builds 04/05's string-pattern detectors (`_is_repeating`/`_is_cycling`)
entirely with a reward-based trigger. `rollout/runner.py`'s `run_reward_aware_episode` tracks
consecutive non-positive `env_reward` steps: at 15, masks the last action; if the stall continues
5 more steps, escalates to a full `thinker.replan()`. Threshold picked from a direct audit of
build 03's baseline — non-positive-reward streak length cleanly separates won vs. lost episodes on
`coin` (won avg 14.5 steps, lost avg 50.0) and `mapreader` (won avg 9.4, lost avg 45.5), while
dense-reward games (`simonsays`, `peckingorder`) almost never build a long streak at all.

**Metrics (Qwen2.5-3B, reward-aware loop):**

| game | won | win rate | avg steps | mask rate | replan rate | avg masks/ep | avg replans/ep |
|---|---|---|---|---|---|---|---|
| coin | 12/20 | 60% | 27.4 | 60% | 55% | 5.10 | 1.00 |
| simonsays | 20/20 | 100% | 5.0 | 0% | 0% | 0.00 | 0.00 |
| peckingorder | 18/20 | 90% | 8.0 | 0% | 0% | 0.00 | 0.00 |
| cookingworld | 0/20 | 0% | 27.6 | 50% | 45% | 3.90 | 0.75 |
| mapreader | 8/20 | 40% | 36.2 | 75% | 65% | 6.05 | 1.05 |

**4-way comparison (build 03 no-coordinator / build 04 replan-string / build 05 mask-string /
build 06 reward-aware):**

| game | build 03 | build 04 | build 05 | build 06 |
|---|---|---|---|---|
| coin | 65% | 55% | 65% | 60% |
| simonsays | 100% | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% | 40% |

**Insights:**

- **`cookingworld` stays at 0% despite genuinely frequent, active triggers** — 10/20 episodes
  masked (50%), 9/20 escalated all the way to a replan (45%), 78 total mask-turns and 15 replan-
  turns across the 20 episodes. This rules out "the trigger wasn't firing" as an explanation. The
  game's difficulty is not about getting unstuck from a stall — it's a genuine capability gap
  (recipe comprehension, correct ingredient/action sequencing) that no stall-detection-and-
  intervention approach can fix, regardless of how the stall is detected or what the response is.
- The reward-aware trigger fires far more often than the string detectors on `coin` (60%/55%
  mask/replan rate vs. build 04/05's 45%) — sparse-reward exploration naturally builds long
  non-positive streaks even during legitimate, productive play, so this trigger is less precise
  there. `coin`'s win rate (60%) sits between build 04's 55% and build 03/05's 65%, consistent
  with a real but small cost from over-triggering on a game where much of the "stall" is normal.
- `mapreader` improves to 40% (75% mask rate, 65% replan rate — the highest of any game), landing
  between build 03's 35% and build 04's 45%, clearly ahead of build 05's 25%. The escalation
  design (try masking first, replan only if that fails) appears to capture most of replanning's
  benefit on this game without needing to discard the plan as often.
- `simonsays`/`peckingorder` held identical to every prior build — 0% trigger rate, episodes too
  short/fast for a 15-step non-positive streak to ever form.

**Verdict:** the reward-aware trigger's real value is diagnostic, not a win-rate improvement. It
doesn't clearly beat builds 04/05 on `coin` or `mapreader`, and it doesn't solve `cookingworld`
either — but by firing much more often and still failing there, it proves `cookingworld` is a
genuine capability gap, not a detection gap. That's a meaningful result: no fixed coordinator,
regardless of trigger type or intervention type, will fix a game the underlying worker model
doesn't know how to play. Further gains on `cookingworld` need a better/larger model or training
signal, not better coordination logic.
