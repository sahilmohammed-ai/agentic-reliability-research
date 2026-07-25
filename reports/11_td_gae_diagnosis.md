# TWX 11 — TD/GAE Labels & Failure-Diversity Diagnosis

**Date:** 2026-07-23

**Change:** tested whether AgentPRM-style TD/GAE labeling beats build 10's Monte-Carlo labeling,
then diagnosed *why* the verifier's quality is capped. Larger collection (1,200 episodes, 4 games,
`cookingworld` dropped), relabeled with a trained-model value function, retrained, evaluated on the
same held-out set as build 10.

**Why TD/GAE needed build 10 first:** MC return-to-go on sparse-reward games is mostly positional
decay (`gamma^steps-to-goal`), barely encoding *which action was good*. TD/GAE's residual
`delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)` is action-dependent, but only if `V(s)` is an
*imperfect* estimate — an exact MC return makes every residual identically zero by construction (a
real failed first attempt confirmed this). Build 10's checkpoint is exactly the imperfect `V(s)`
needed, so `rollout/label_td.py` bootstraps off it.

**Training (`train_v4.log`):** val loss bottomed at epoch 2 (0.060 combined) then rose at epochs
3-4 while train loss kept dropping — classic overfitting. Best-val checkpoint saved separately, so
`verifier.pt` is the epoch-2 weights regardless. **2 epochs is optimal at this data scale.**

**Held-out evaluation, TD/GAE (v4) vs. MC (v1):**

| game | v1 (MC) AUC | v4 (TD/GAE) AUC |
|---|---|---|
| coin | 0.569 | 0.592 |
| peckingorder | 0.714* | 0.612* |
| mapreader | 0.798 | 0.767 |
| overall gap (4 games, no cookingworld) | +0.191 | +0.133 |

\* `peckingorder` has only 5 lost turns in eval — statistically meaningless (noise).

**Insights:**

- **TD/GAE did not beat MC** — a near-tie on the games with real failure data (v4 edges `coin`, v1
  edges `mapreader`), dragged apart in aggregate only by `peckingorder`'s unmeasurable 5-sample set.
  The extra complexity didn't earn its keep at this data scale.
- **AUC exposed the real problem: failure diversity, not labeling.** The verifier discriminates
  well only where both outcomes are well-represented (`mapreader`, AUC ~0.78); `coin` has losses but
  weak per-turn signal (near-chance); `simonsays`/`peckingorder` are unmeasurable (≈0 and 5 lost
  turns). A verifier learns discrimination from contrast, and most games don't provide it.

**Verdict:** labeling scheme is not the bottleneck — MC is marginally simpler and no worse. The
ceiling is **failure diversity**: too many games a 3B worker almost always wins. Motivates build 12:
difficulty-calibrate the games (`scripts/difficulty_sweep.py`) so every one produces balanced
win/loss outcomes, then retrain on genuinely balanced data.
