# TWX 02 — Model Baselines

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

| game | Sonnet 5 (build 01, unstratified) | Opus 4.8 | GPT-5.4 | Qwen2.5-3B |
|---|---|---|---|---|
| coin | 83% (n=18) | 40% | 40% | 55% |
| simonsays | 97% (n=32) | 85% | 5% | 55% |
| peckingorder | untested | 100% | 100% | 45% |
| cookingworld | 0% (n=10) | 0% | 0% | 0% |
| mapreader | 31% (n=13) | 60% | 25% | 40% |

## Insights

- `cookingworld` is unsolved by every model tested, including Qwen2.5-3B (0% for Sonnet 5, Opus
  4.8, GPT-5.4, and Qwen2.5-3B) but not broken — 1-5% nonzero-reward turns across all four show
  real mid-episode partial credit exists.
- GPT-5.4 has a striking, specific weakness on `simonsays` (5% win) — a game Opus 4.8, Sonnet 5,
  and Qwen2.5-3B all treat as at least moderately solvable (85%, 97%, 55%). At n=20 with a gap
  this large, this reads as a genuine capability gap, not noise.
- `peckingorder` is NOT universally trivial — it's trivial only at frontier-model scale. Opus 4.8
  and GPT-5.4 both hit 100% (always exactly 8 steps), but Qwen2.5-3B drops to 45%, with much more
  variable episode length (1-21 steps). Revises the earlier read of this game as
  "deterministic/no failure signal" — that only holds for frontier-scale models.
- Qwen2.5-3B, despite being far smaller and run locally, beats both frontier API models on `coin`
  (55% vs 40%/40%) and beats GPT-5.4 badly on `simonsays` (55% vs 5%, though still behind Opus's
  85%). Model size alone doesn't predict per-game performance in this game mix.
- No model dominates across the board. Model choice for this project should stay game/task-aware
  rather than picking a single "best" model.
