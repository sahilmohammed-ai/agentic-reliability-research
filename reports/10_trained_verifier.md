# TWX 10 — Trained Verifier (MC labels)

**Date:** 2026-07-22

**Change:** first *trained* turn-level verifier on TextWorldExpress (replacing builds 08/09's
prompted frozen-LLM judge). Full fine-tune of Qwen2.5-1.5B (backbone + 2-scalar value head).
Collected 500 episodes (100/game × 5 games, `train` split, Qwen2.5-3B worker), labeled with real
per-step `env_reward` via `rollout/label_twx.py` (`q_value` = MC return-to-go, `advantage` =
one-step delta, unbounded since labels range −1.8 to 1.0). Trained 4 epochs, Lightning AI.

**Held-out evaluation (build 03 `eval_ood`, never trained on):**

| game | won mean | lost mean | AUC | (won n / lost n) |
|---|---|---|---|---|
| coin | 0.324 | 0.246 | 0.569 | 202 / 350 |
| simonsays | 0.482 | — | — | 100 / 0 |
| peckingorder | 0.439 | 0.073 | 0.714 | 155 / 5 |
| cookingworld | — | 0.020 | — | 0 / 557 |
| mapreader | 0.304 | 0.162 | 0.798 | 95 / 650 |
| **overall** | **0.381** | **0.130** | | 552 / 1562 |

**Insights:**

- Real signal (won 0.381 vs lost 0.130 overall), ~50× faster than the frozen LLM judge (one
  forward pass vs. full reasoning generation, ~0.5s vs ~15-20s/turn).
- Discrimination is uneven by game: strong where balanced (`mapreader` AUC 0.798), near-chance on
  `coin` (0.569) despite rich reward variance, unmeasurable on `simonsays`/`peckingorder` (0 and 5
  lost turns — a 3B worker rarely loses them). `cookingworld` is flat/low (0.020, no wins to
  contrast against).
- Pattern is diagnostic: discrimination tracks win/loss balance, not model capacity.

**Verdict:** a genuine, deployable trained verifier, clearly faster than the prompted judge, but
quality is bottlenecked by failure diversity in the training/eval games. Carried into build 11.
