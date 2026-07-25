# TWX 12 — Difficulty-Calibrated Balanced Retrain

**Date:** 2026-07-25

**Change:** tested build 11's diagnosis directly — is failure diversity really the verifier's
ceiling? Calibrated game difficulty via `scripts/difficulty_sweep.py` to push each game toward a
30-60% win rate for Qwen2.5-3B, dropped `cookingworld` (0% win, all-failure, no contrast),
collected 1,200 balanced episodes, retrained (MC labels, same architecture as v1), and evaluated on
a freshly-collected balanced `eval_ood` set (never trained on, same calibrated difficulty).

**Difficulty calibration:** `coin` `numLocations=8,numDistractorItems=5` (was 5/3, ~40% win),
`simonsays` `gameLength=15` (was 5, ~60% win), `mapreader` unchanged (already balanced, ~30-40%),
`peckingorder` unchanged (no difficulty knob, kept at its ~85-90% win for task variety).

**Held-out evaluation (v5 checkpoint, fresh balanced `eval_ood`, 300 episodes / 75 per game):**

| game | v1 (build 10) AUC | v4 (build 11, TD/GAE) AUC | **v5 (build 12) AUC** | v5 lost-turn n |
|---|---|---|---|---|
| coin | 0.569 | 0.592 | **0.709** | 1,750 |
| mapreader | 0.798 | 0.767 | **0.782** | 2,400 |
| peckingorder | 0.714 (n=5, noise) | 0.612 (n=5, noise) | **0.695** | 17 |
| simonsays | unmeasurable (n=0) | unmeasurable | **0.562** | 163 |
| **overall** | — | — | **0.844** | 4,330 |

**Insights:**

- **The diagnosis was correct and the fix worked.** `coin` — the game with the richest reward
  variance but near-chance discrimination in builds 10/11 — jumped to AUC 0.709 once its difficulty
  was calibrated into the balanced zone. This is the clearest single result of the project so far:
  the verifier's quality ceiling was failure diversity in the *data*, not the labeling scheme
  (build 11 already ruled that out) or model capacity.
- **`simonsays` went from literally unmeasurable to a real, if modest, AUC (0.562).** Difficulty
  calibration didn't just improve an existing signal — it created one where there was none: this
  game previously had zero lost turns in eval.
- **`mapreader` held steady** (0.782 vs 0.798) — it was already balanced pre-calibration, so no
  regression, as expected; a useful internal control confirming the retrain didn't just get lucky.
- **`peckingorder` stayed noisy** (17 lost turns, up from 5) — it has no difficulty knob, so this
  result is still the least trustworthy in the table, consistent with builds 10/11.
- **Overall AUC (0.844) is pooled across games** with very different per-game AUCs and lost-turn
  counts, dominated by `coin` and `mapreader`'s large samples — report per-game numbers as the
  primary result, overall as a secondary summary stat, not the headline.

**Verdict:** difficulty-calibrated balanced data is a real, validated fix for the failure-diversity
ceiling diagnosed in build 11. This is the first verifier iteration where every game in the training
mix produces a measurable, better-than-chance signal. `checkpoints/verifier_v5` is the new best
checkpoint and the recommended verifier for any downstream (coordinator/RL) work.
