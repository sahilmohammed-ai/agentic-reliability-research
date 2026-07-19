# TWX 02 — Agentic Loop Baselines

**Date:** 2026-07-18

**Environment:** TextWorldExpress, zero-coordination baseline (thinker plans once, worker acts
until done/cap, no replanning/masking/verifier). `--split eval_ood`. 50-step cap. Stratified: 20
episodes per game, `TRAINING_GAMES` mix (`coin`, `simonsays`, `peckingorder`, `cookingworld`,
`mapreader`).

## Opus 4.8

Thinker + Worker both `claude-opus-4-8`.

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 8/20 | 40% | 31.6 | 1.3% |
| simonsays | 17/20 | 85% | 4.8 | 100.0% |
| peckingorder | 20/20 | 100% | 8.0 | 50.0% |
| cookingworld | 0/20 | 0% | 24.4 | 4.7% |
| mapreader | 12/20 | 60% | 25.0 | 6.2% |

## GPT-5.4

Thinker + Worker both `gpt-5.4`.

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 8/20 | 40% | 33.1 | 1.2% |
| simonsays | 1/20 | 5% | 2.9 | 100.0% |
| peckingorder | 20/20 | 100% | 8.0 | 50.0% |
| cookingworld | 0/20 | 0% | 22.4 | 3.6% |
| mapreader | 5/20 | 25% | 38.9 | 1.5% |

## Qwen2.5-3B-Instruct (HF, local)

Thinker + Worker both `hf:Qwen/Qwen2.5-3B-Instruct`.

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 11/20 | 55% | 29.8 | 1.8% |
| simonsays | 11/20 | 55% | 4.2 | 100.0% |
| peckingorder | 9/20 | 45% | 6.2 | 48.0% |
| cookingworld | 0/20 | 0% | 36.6 | 3.4% |
| mapreader | 8/20 | 40% | 37.3 | 2.8% |

## Cross-model comparison

| game | Opus 4.8 | GPT-5.4 | Qwen2.5-3B |
|---|---|---|---|
| coin | 40% | 40% | 55% |
| simonsays | 85% | 5% | 55% |
| peckingorder | 100% | 100% | 45% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 60% | 25% | 40% |

## Insights

- `cookingworld` is unsolved by every model tested (0%) but not broken — 1-5% nonzero-reward
  turns across all three show real mid-episode partial credit exists.
- GPT-5.4 has a striking, specific weakness on `simonsays` (5% win) vs. Opus 4.8 and Qwen2.5-3B
  (85%, 55%) — a genuine capability gap, not noise.
- `peckingorder` is trivial only at frontier-model scale (Opus 4.8/GPT-5.4 both 100%);
  Qwen2.5-3B drops to 45%.
- No model dominates across the board. Model choice should stay game/task-aware.

## Single-agent vs. agentic loop (build 01 vs. build 02, same models/games)

| game | Opus 4.8 (single → loop) | GPT-5.4 (single → loop) | Qwen2.5-3B (single → loop) |
|---|---|---|---|
| coin | 85% → 40% | 50% → 40% | 60% → 55% |
| simonsays | 100% → 85% | 100% → 5% | 100% → 55% |
| peckingorder | 100% → 100% | 95% → 100% | 40% → 45% |
| cookingworld | 0% → 0% | 0% → 0% | 0% → 0% |
| mapreader | 50% → 60% | 35% → 25% | 30% → 40% |

The thinker+worker agentic loop underperforms the single-agent baseline (`01_model_baselines.md`)
on `simonsays` for every model, most dramatically GPT-5.4 (100% → 5%) — the upfront plan step
actively hurts there. `mapreader` is the one game where the agentic loop tends to do better,
plausibly because it rewards following a multi-step navigation instruction. Any future
verifier/coordinator layer should be benchmarked against the single-agent baseline, not assumed
to need the thinker+worker split as its foundation.
