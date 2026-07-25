# TWX 11 — TD/GAE Labels & Failure-Diversity Diagnosis

**Date:** 2026-07-23

**Change:** tested whether AgentPRM-style TD/GAE labeling (the "theoretically correct" method)
beats build 10's simpler Monte-Carlo labeling, then diagnosed *why* the verifier's quality is
capped. Larger balanced-ish collection (1,200 episodes, 4 games, `cookingworld` dropped), relabeled
with a trained-model value function, retrained, evaluated on the same held-out set.

**Why TD/GAE, and why it needed build 10 first.** MC return-to-go on sparse-reward games is mostly
positional decay (`gamma^steps-to-goal`) — it barely encodes *which action was good*, only *how
close to the end* the turn is. TD/GAE's residual `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)` is
action-dependent, but only if `V(s)` is an *imperfect* estimate the reward can disagree with. An
exact MC return makes every residual identically zero (it satisfies Bellman by construction — a
real failed first attempt confirmed this). Build 10's trained verifier is exactly the imperfect
`V(s)` needed, so `rollout/label_td.py` uses it to bootstrap genuine, nonzero residuals — the
train-freeze-relabel iteration this project's prior-art research found real implementations use.

**Training (`train_v4.log`).** The loss curve — captured for the first time — settled a standing
question: val loss bottomed at **epoch 2** (0.060 combined) then rose at epochs 3-4 while train
loss kept dropping. Classic overfitting after epoch 2. The best-val checkpoint is saved separately,
so `verifier.pt` is the epoch-2 weights regardless. **2 epochs is optimal at this data scale**; 4
was too many.

**Held-out evaluation, TD/GAE (v4) vs. MC (v1):**

| game | v1 (MC) AUC | v4 (TD/GAE) AUC |
|---|---|---|
| coin | 0.569 | 0.592 |
| peckingorder | 0.714* | 0.612* |
| mapreader | 0.798 | 0.767 |
| overall gap (4 games, no cookingworld) | +0.191 | +0.133 |

\* `peckingorder` has only 5 lost turns in eval — its AUC/gap is statistically meaningless (noise).

**Insights:**

- **TD/GAE did not beat MC.** On the games with real, measurable failure data (`coin`, `mapreader`),
  it's a near-tie — v4 edges `coin` (0.592 vs 0.569, the game the diagnosis predicted TD/GAE would
  most help), v1 edges `mapreader`. The aggregate 4-game gap slightly favors MC (+0.191 vs +0.133),
  but that's dragged by `peckingorder`'s unmeasurable 5-sample loss set. Net: a wash, not a clear
  win for either. The extra TD/GAE complexity did not earn its keep at this data scale.
- **AUC exposed the real problem — failure diversity, not labeling.** Switching from mean-gap to AUC
  (scale-invariant separation) showed the verifier only discriminates well where both outcomes are
  well-represented: `mapreader` (~30-40% win, AUC ~0.78) works; `coin` (near-chance ~0.57) has
  losses but weak per-turn label signal; `simonsays`/`peckingorder` are unmeasurable (≈0 and 5 lost
  turns). A verifier learns discrimination from *contrast*, and most games don't provide it.
- **Side note (`cookingworld`):** v4 scores it strongly negative despite never training on it —
  real failure-recognition generalization — but it's dropped from the pipeline, so this is academic.

**Verdict:** the labeling scheme (MC vs. TD/GAE) is not the bottleneck — both are comparable, MC
marginally simpler and no worse. The verifier's quality ceiling is **failure diversity**: too many
games a 3B worker almost always wins, so there are too few failure examples to learn from or
evaluate on. This directly motivates build 12: difficulty-calibrate the games so every one produces
balanced win/loss outcomes (via `scripts/difficulty_sweep.py`), then retrain on genuinely balanced
data.
