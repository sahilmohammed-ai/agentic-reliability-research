# Build 9

**Date:** 2026-07-11

**Overview:** First run in a second environment, TextWorldExpress, to test whether the setup and the masking result generalize beyond ALFWorld. Same config as build_8 (mask mode, HF Qwen2.5-3B) with only the environment changed. Also the first build with per-turn token/cost tracking. An earlier ScienceWorld attempt was abandoned (0% win, non-viable substrate; data kept under data/rollouts/scienceworld_abandoned), so TextWorldExpress takes the build_9 slot.

**Environment:**
- TextWorldExpress (coin, simonsays, peckingorder games; sampled at random per episode)
- Games chosen because TALES benchmark shows a ~3B model solves them 80-100%; deliberately excludes the "sorting" game (instant-death on a wrong move, same trap that made ScienceWorld unusable)
- Train split, 50-step cap

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, HF Transformers): plan at episode start, replans on stagnation
- Worker (Qwen2.5-3B-Instruct, HF Transformers): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed, "mask" mode (stagnation replan + single-turn masking of the most-repeated action), identical to build_8

**Metrics:**
- 100 episodes total
- 81 won, 19 lost (81% win rate)
- Won episodes averaged 10.5 steps; lost averaged 21.7
- 89 mask events fired
- Tokens: 456,448 prompt + 17,318 completion = 473,766 total, avg ~4,738 tokens/episode
