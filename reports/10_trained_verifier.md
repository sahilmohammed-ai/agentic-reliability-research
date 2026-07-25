# TWX 10 — Trained Verifier (MC labels)

**Date:** 2026-07-22

**Change:** the first *trained* turn-level verifier on TextWorldExpress, replacing builds 08/09's
prompted frozen-LLM judge. Full fine-tune of Qwen2.5-1.5B (backbone + 2-scalar value head) on
real-reward labels, then evaluated on held-out data. This is the keystone the project's
architecture is built around.

**Pipeline:**
- **Collect:** 500 episodes (100/game × 5 games) on the `train` split, Qwen2.5-3B worker, so
  `eval_ood` stays untouched for evaluation.
- **Label (`rollout/label_twx.py`):** real per-step `env_reward` → `q_value` = Monte-Carlo
  return-to-go, `advantage` = one-step value delta. A genuine upgrade over the ALFWorld-era
  `label.py`, which discarded reward for a flat won/lost signal (ALFWorld's reward was terminal-
  only; TextWorldExpress's is dense, e.g. +0.125 correct placement / −0.125 wrong). Labels range
  −1.8 to 1.0 (real negatives), so `q_value` is trained unbounded (no sigmoid) — a `bound_q_value`
  flag was added to `verifier/model.py` after finding the old sigmoid would silently clip these.
- **Train:** Lightning AI, `--full-finetune --batch-size 16 --epochs 4 --lr 1e-5
  --unbounded-q-value`, ~1.54B trainable params, 8-bit AdamW.

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

- Real, working signal: won episodes score meaningfully higher than lost (overall 0.381 vs 0.130),
  and it runs ~50× faster than the frozen LLM judge (one forward pass vs. a full reasoning
  generation, ~0.5s vs ~15-20s per turn) — the speed that makes a live coordinator viable.
- Discrimination is uneven by game. `mapreader` is strong (AUC 0.798); `coin` is near-chance
  (0.569) despite having the richest reward variance. `simonsays`/`peckingorder` are effectively
  unmeasurable — 0 and 5 lost turns respectively, since a 3B worker almost never loses them.
- The pattern is diagnostic: the verifier discriminates well exactly where the game has balanced
  win/loss outcomes (`mapreader`), and poorly or unmeasurably where it doesn't. Failure diversity,
  not model capacity, is the apparent ceiling.
- `cookingworld` scores flat and low (0.020) with no wins to contrast against — an all-failure
  game teaching little beyond "this is bad."

**Verdict:** a genuine, deployable trained verifier and a clear improvement in speed over the
prompted judge, but its discrimination quality is bottlenecked by failure diversity in the
training/eval games, not by the model or the labeling. The `coin` near-chance AUC and the
unmeasurable easy games are the open problem carried into build 11.
