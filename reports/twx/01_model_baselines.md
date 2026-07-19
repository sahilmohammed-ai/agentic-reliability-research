# TWX 01 — Model Baselines (Single-Agent)

**Date:** 2026-07-18

**Architecture:** single-agent baseline — one model, one call per step. Each step, the model sees
the task goal, current observation, admissible commands, and recent action history, and picks one
action directly (`agents/single_agent.py`). No upfront plan, no second model, no coordination.
Distinct from build 02's thinker+worker agentic loop (`agents/thinker.py` plans once,
`agents/worker.py` executes against that plan every step).

**Environment:** TextWorldExpress, `--split eval_ood`, 50-step cap. Stratified: 20 episodes per
game, `TRAINING_GAMES` mix (`coin`, `simonsays`, `peckingorder`, `cookingworld`, `mapreader`).

## Opus 4.8

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 17/20 | 85% | 15.0 | 5.7% |
| simonsays | 20/20 | 100% | 5.0 | 100.0% |
| peckingorder | 20/20 | 100% | 8.0 | 50.0% |
| cookingworld | 0/20 | 0% | 35.1 | 6.4% |
| mapreader | 10/20 | 50% | 32.1 | 4.0% |

## GPT-5.4

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 10/20 | 50% | 26.9 | 1.9% |
| simonsays | 20/20 | 100% | 5.0 | 100.0% |
| peckingorder | 19/20 | 95% | 8.6 | 51.7% |
| cookingworld | 0/20 | 0% | 20.9 | 7.4% |
| mapreader | 7/20 | 35% | 37.0 | 2.3% |

## Qwen2.5-3B-Instruct (HF, local)

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 12/20 | 60% | 28.0 | 2.1% |
| simonsays | 20/20 | 100% | 5.0 | 100.0% |
| peckingorder | 8/20 | 40% | 8.2 | 49.1% |
| cookingworld | 0/20 | 0% | 32.4 | 4.8% |
| mapreader | 6/20 | 30% | 39.4 | 2.4% |

## Cross-model comparison (single-agent)

| game | Opus 4.8 | GPT-5.4 | Qwen2.5-3B |
|---|---|---|---|
| coin | 85% | 50% | 60% |
| simonsays | 100% | 100% | 100% |
| peckingorder | 100% | 95% | 40% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 50% | 35% | 30% |

## Insights

- `simonsays` is fully solved by every model single-agent (100% across Opus, GPT-5.4,
  Qwen2.5-3B).
- `cookingworld` stays unsolved (0%) across every model — the hardest game in the mix.
- `peckingorder` stays near-solved for the frontier models (Opus 100%, GPT-5.4 95%), but
  Qwen2.5-3B is notably weaker (40%) — a real small-model capability gap.
- No model dominates across the board; `coin` and `mapreader` show real spread by model.
