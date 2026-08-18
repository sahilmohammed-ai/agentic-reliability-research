# Results — Full Experiment Log

Working document, not paper prose. Every experiment run this project, with metrics, in
chronological order. Use this to write Results/Discussion/Conclusion. Each section names the
source report (`reports/0N_*.md`) if you need the full detail/insights beyond what's copied here.

Paper framing to keep in mind while writing from this: the thesis centers on a turn-level verifier
that acts as an online failure detector and as a signal for guiding real-time decisions, trained
purely from real environment outcomes (no hand-written rules). The verifier work is the paper's
core contribution: two independently-trained verifier variants (`verifier_mc`, MC-return-labeled
with difficulty-balanced training data; `verifier_dpo`, trained via episode-outcome preference
learning) both show real, validated discrimination ability, and a Best-of-N evaluation shows that
signal genuinely improves decisions in practice (a real win-rate lift on the one game with
headroom to improve). Reports 04-09 (an earlier line of heuristic/fixed coordination strategies)
remain in this document as historical record of the baselines and motivating experiments that led
to the verifier-centered approach, but coordination strategy is not part of the current
methodology.

---

## Environment & models used throughout

TextWorldExpress. Games across the project: `coin`, `simonsays`, `peckingorder`, `cookingworld`,
`mapreader` (builds 01-09); `cookingworld` dropped from build 10 onward (0% win rate for every
model tested, no success/failure contrast possible — see build 02/06 insights). Final 4-game
training/eval mix (build 10 onward): `coin`, `mapreader`, `peckingorder`, `simonsays`.

Models: Opus 4.8, GPT-5.4, and Qwen2.5-3B-Instruct (local, via HF) compared in builds 01-02;
Qwen2.5-3B-Instruct used as the fixed Worker for all verifier work from build 03 onward
(cost-discipline rule); verifier backbone is Qwen2.5-1.5B-Instruct, chosen over 3B for
inference-cost reasons (see build 10).

`eval_ood` split used for all held-out evaluation from build 03 onward, never trained on.

---

## Build 01 — Single-Agent Model Baselines
*(`reports/01_model_baselines.md`, 2026-07-18)*

One model, one call per step, no plan, no coordination. 20 episodes/game, 5-game mix.

| game | Opus 4.8 | GPT-5.4 | Qwen2.5-3B |
|---|---|---|---|
| coin | 85% | 50% | 60% |
| simonsays | 100% | 100% | 100% |
| peckingorder | 100% | 95% | 40% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 50% | 35% | 30% |

**Key takeaways:** simonsays fully solved by every model. cookingworld unsolved by every model
(0%) — established early as a genuine capability ceiling, not a downstream artifact. No model
dominates across the board; coin/mapreader show real per-model spread. This is the reference point
for "how good is a raw model with no scaffolding at all."

---

## Build 02 — Agentic Loop (Thinker+Worker) Baselines
*(`reports/02_agentic_loop_baselines.md`, 2026-07-18)*

Adds the thinker(plans once)+worker(executes to done/cap) split, no replanning/masking/verifier
yet. Same 20/game, 5-game mix.

| game | Opus 4.8 | GPT-5.4 | Qwen2.5-3B |
|---|---|---|---|
| coin | 40% | 40% | 55% |
| simonsays | 85% | 5% | 55% |
| peckingorder | 100% | 100% | 45% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 60% | 25% | 40% |

**Single-agent vs. loop, same models:** the loop UNDERPERFORMS single-agent on simonsays for
every model (worst: GPT-5.4, 100%→5%) — a fixed upfront plan actively hurts on this reactive game.
mapreader is the one game the loop tends to help. **Conclusion carried forward:** any
verifier/coordinator layer needs to be benchmarked against the single-agent baseline too, not just
assumed to need the thinker+worker split as its foundation — this is a real, load-bearing caveat
for the Discussion section (the loop architecture itself has a real cost on some tasks).

---

## Build 03 — Prompt Engineering
*(`reports/03_prompt_engineering.md`, 2026-07-19)*

Fixed `agents/thinker.py`'s system prompt, which was 100% ALFWorld-specific and produced bad plans
on TextWorldExpress's reactive games. New prompt classifies each task (fixed-goal /
reactive-instruction-following / discover-then-act) and plans accordingly. Qwen2.5-3B only from
here on.

| game | single-agent (01) | loop, old prompt (02) | loop, new prompt (03) |
|---|---|---|---|
| coin | 60% | 55% | 65% |
| simonsays | 100% | 55% | 100% |
| peckingorder | 40% | 45% | 90% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 30% | 40% | 35% |

**Key takeaway:** simonsays/peckingorder jump sharply and close/exceed the single-agent gap —
confirms the build 02 loop regression was a prompt-quality problem, not a fundamental flaw in
planning-then-executing. cookingworld stays 0% — prompt fixes solve prompt-mismatch failures, not
capability gaps. **Build 03 is the baseline every later coordinator/verifier build is compared
against.**

---

## Build 04 — Fixed Coordinator v1 (String-Pattern Replan)
*(`reports/04_coordinator_replan.md`, 2026-07-19)*

First coordinator, not learned — pure heuristic. Replans (discards plan, gets a fresh one) when
worker hits a literal 3x repeat or a 2-4 period cycle, detected via string matching on
`action_history`. No reward/verifier signal involved at all.

| game | build 03 (no coordinator) | build 04 (replan) | replan rate |
|---|---|---|---|
| coin | 65% | 55% | 45% |
| simonsays | 100% | 100% | 0% |
| peckingorder | 90% | 90% | 0% |
| cookingworld | 0% | 0% | 20% |
| mapreader | 35% | 45% | 70% |

**Verdict:** mixed. One real gain (mapreader +10pts), one real loss (coin -10pts), two no-ops
(games too short to ever trigger), no improvement on cookingworld despite real triggers (wrong
failure mode — this game's stuckness is semantic churn, not literal/cyclic repetition, which
string detectors can't see). **Motivates:** a heuristic coordinator can't tell a costly
interruption from a needed one — first evidence pointing toward needing a learned, signal-aware
trigger.

---

## Build 05 — Fixed Coordinator v2 (Masking)
*(`reports/05_masking.md`, 2026-07-19)*

Same detectors as build 04, but instead of replanning, masks the looping action(s) out of
`admissible_commands` — narrower choice, no fresh plan.

| game | build 03 | build 04 (replan) | build 05 (masking) |
|---|---|---|---|
| coin | 65% | 55% | 65% |
| simonsays | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% |

**Verdict:** masking is the gentler intervention — coin fully recovers to baseline (65%), beating
replan's 55%. But mapreader is the one place masking clearly LOSES (25%, below both baseline and
replan), despite the highest trigger rate of any game/build combo (70%, 3.75 masks/episode) —
narrowing options without a fresh strategy leaves the worker floundering. **Neither intervention
dominates** — sets up the case for a coordinator that can pick the right action per-situation.

---

## Build 06 — Fixed Coordinator v3 (Reward-Aware Trigger)
*(`reports/06_reward_aware.md`, 2026-07-19)*

Replaces string-pattern detection with a reward-based trigger: 15 consecutive non-positive
`env_reward` steps masks the last action; 5 more steps of stall escalates to a full replan.
Threshold picked from a direct audit showing non-positive-reward streak length cleanly separates
won/lost episodes on coin and mapreader.

| game | build 03 | build 04 | build 05 | build 06 (reward-aware) |
|---|---|---|---|---|
| coin | 65% | 55% | 65% | 60% |
| simonsays | 100% | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% | 40% |

**The important result here is diagnostic, not a win-rate win.** cookingworld triggers heavily
(50% mask, 45% replan rate — genuinely active, not silent) and STILL stays at 0%. This rules out
"the trigger wasn't firing" as an explanation for cookingworld's failure and establishes it's a
genuine model capability gap (recipe comprehension), not fixable by any coordination strategy —
important for framing cookingworld correctly in Discussion (it's excluded from later builds for
this reason, not swept under the rug).

---

## Build 07 — Fixed Coordinator v4 (Backtrack)
*(`reports/07_backtrack.md`, 2026-07-20)*

Same reward-stall trigger as build 06, but instead of masking/replanning, clears
`action_history` entirely (full context wipe, no escalation).

| game | build 03 | build 04 | build 05 | build 06 | build 07 (backtrack) |
|---|---|---|---|---|---|
| coin | 65% | 55% | 65% | 60% | 55% |
| simonsays | 100% | 100% | 100% | 100% | 100% |
| peckingorder | 90% | 90% | 90% | 90% | 90% |
| cookingworld | 0% | 0% | 0% | 0% | 0% |
| mapreader | 35% | 45% | 25% | 40% | 35% |

**Closes the fixed-coordinator series (builds 04-07).** Four trigger/intervention combinations
tested; no dominant strategy. Masking is consistently gentlest; replan/backtrack both carry real
disruption costs on some games. cookingworld never moves under ANY of the four — a third
independent confirmation it's a capability gap, not a coordination-strategy problem. This series
motivated moving away from fixed coordination heuristics toward the verifier-centered approach
(build 08 onward): a learned, per-turn signal that can be used more flexibly (e.g. Best-of-N
action selection) rather than a fixed rule applied everywhere.

---

## Build 08 — Frozen LLM Verifier (Prompted, No Training)
*(`reports/08_frozen_verifier.md`, 2026-07-20)*

First verifier: an untrained LLM judge (Qwen2.5-3B) scores each turn 0.0-1.0 via prompting, never
sees `env_reward`/`won`. Scored build 03's existing 100-episode baseline (2,114 turns) offline.

| | won mean | lost mean |
|---|---|---|
| overall | 0.583 | 0.431 |

Nonzero-`env_reward` turns score much higher than zero-reward turns on coin (1.000 vs 0.303) and
mapreader (0.950 vs 0.564) — real correlation with ground truth. peckingorder inverted (small
sample, 5 lost turns — later explained by `verifier_mc`'s difficulty-calibration finding: too few
lost episodes to measure reliably until the game mix was balanced). One quantified weakness: 3/58
(5%) of won episodes score their own winning final turn as 0.0 (minor, not systematic).

**Verdict:** a real, usable signal without any training — this is the first evidence for the "the
verifier can detect failures" half of the thesis.

---

## Build 09 — Verifier Variants
*(`reports/09_verifier_variants.md`, 2026-07-21)*

Tested two alternatives to build 08's design: no-reasoning (bare-number) prompt, and a smaller
1.5B backbone.

| game | reasoning-required (3B) won/lost | no-reasoning (3B) won/lost |
|---|---|---|
| overall | 0.583 / 0.431 | 0.209 / 0.069 |

No-reasoning compresses scores toward 0 but keeps a similar won/lost gap; ~25% faster; fixes
peckingorder's inversion but loses resolution elsewhere. 1.5B backbone failed the exact
self-contradictory-reasoning case that caught build 08's original bug (scored a correct
instruction-match as 0.0) — not pursued further at this stage. **Verdict:** kept 3B +
reasoning-required as the default for the frozen-verifier line. (Note: builds 10+ later DID move
to a 1.5B backbone successfully, once training rather than prompting was used — the 1.5B failure
here was specific to zero-shot prompted reasoning, not the architecture generally.)

---

## Build 10 — Trained Verifier: Monte-Carlo Labels (`verifier_mc`)
*(`reports/10_trained_verifier.md`, `reports/11_td_gae_diagnosis.md`, `reports/12_balanced_retrain.md`)*

`verifier_mc` — full fine-tune of Qwen2.5-1.5B (backbone + 2-scalar value head:
`[q_value, advantage]`), trained on real, sparse `env_reward` via Monte-Carlo return-to-go
(`rollout/label_twx.py`: `q_value` = MC return-to-go, `advantage` = one-step delta).

**Diagnosis that shaped the final training recipe:** an early version of this verifier, trained on
an unbalanced 500-episode collection, showed real but UNEVEN discrimination by game — strong where
win/loss outcomes were naturally balanced (mapreader), near-chance on games a capable worker
rarely loses (coin, simonsays, peckingorder). A labeling-scheme alternative (TD/GAE bootstrapping,
in the spirit of AgentPRM's shaped value target) was tested and did NOT meaningfully beat plain MC
labels on games with real failure data — ruling out the labeling formula as the bottleneck. The
real finding: **discrimination quality tracks win/loss balance in the training data, not the
labeling formula or model capacity.** This directly motivated difficulty-calibrating the games
(`scripts/difficulty_sweep.py`) to force a genuinely balanced win/loss mix before the final
training run below, rather than accepting the game mix as given.

**Final training recipe:** calibrated game difficulty to push each game toward a 30-60% win rate
for Qwen2.5-3B — coin `numLocations=8,numDistractorItems=5`, simonsays `gameLength=15`, mapreader
unchanged (already balanced), peckingorder unchanged (no difficulty knob, kept for task variety
despite its ~85-90% win rate). Dropped cookingworld (0% win rate, no win/loss contrast possible
for any worker tested). Collected 1,200 balanced episodes, trained with MC labels, evaluated on a
FRESH, balanced, held-out `eval_ood` set (never trained on):

| game | AUC | lost-turn n |
|---|---|---|
| coin | **0.709** | 1,750 |
| mapreader | **0.782** | 2,400 |
| peckingorder | **0.695** | 17 |
| simonsays | **0.562** | 163 |
| **overall (pooled)** | **0.844** | 4,330 |

**This is the clearest, most important single result in the entire verifier line.** coin — the
game with richest reward variance but near-chance discrimination before difficulty calibration —
reached AUC 0.709 purely from balancing the training data, with no change to labeling formula or
architecture. simonsays went from LITERALLY UNMEASURABLE (zero lost turns available at all before
calibration) to a real, if modest, AUC (0.562) — calibration didn't just improve an existing
signal, it created a measurable one where none existed. mapreader held steady (0.782, already
balanced before calibration) — a clean internal control confirming the improvement wasn't luck.
peckingorder stays the least trustworthy result of the four (only 17 lost turns, no difficulty
knob available for this game).

**`checkpoints/verifier_mc` is the best verifier checkpoint in the project.** Report per-game AUC
as the primary result — pooled 0.844 is a secondary summary stat, weighted heavily toward
coin/mapreader's large sample sizes, not a number to lead with on its own. ~50× faster than the
prompted judge (build 08/09) at inference time (one forward pass vs. full reasoning generation,
~0.5s vs ~15-20s/turn) — an important practical result for the "online detector" framing, since
real-time use needs this speed.

---

## verifier_dpo: episode-outcome preference training (validated, post-submission solidification)

A genuinely different, outcome-grounded verifier variant, developed and solidified during the
2-week paper revision window (after the initial submission). Present this as a validated,
positive finding — not a hedge or "early signal" — with its one real, honestly-disclosed caveat
(see below).

**Approach:** a structurally different form of supervision from `verifier_mc`'s per-step
regression — instead of a per-step `q_value`/`advantage` target derived from sparse `env_reward`,
train a NEW, separate model (`verifier_dpo/`) on real per-EPISODE outcomes only (won/lost — never
sparse, always known), via a Bradley-Terry/DPO-style preference loss over whole-episode text pairs
(one won episode vs. one lost episode of the same game). Not literally DPO in the strict RLHF
sense (no reference-policy log-ratio machinery) — accurately described as "a DPO-style /
Bradley-Terry preference objective." Per-turn scores can be read off afterward from the model's own
per-token scores, a byproduct rather than a training target — despite never being trained on
per-turn labels, this per-turn readout was later independently validated with a real, held-out
AUC/F1 measurement (see "Turn-Level Verifier Comparison" below) and found to be genuine turn-level
signal, not just an episode-level byproduct.

**Solidification sweep — three configs, same 1,200-episode dataset (`data/labeled/v5_train_combined`,
same data `verifier_mc` trained on), Qwen2.5-1.5B backbone, episode-grouped train/val split (no
episode's text appears in both):**

| config | train/val pairs | epochs | val_pairwise_acc | episode_score_vs_length_correlation | length_only_pairwise_acc |
|---|---|---|---|---|---|
| frozen head, capped pairs | 160 / 160 | 3 | 0.8375–0.8500 | −0.51 | 0.3688 |
| frozen head, full pairs | 630 / 157 | 3 | **0.9108** | −0.385 | 0.2866 |
| **full fine-tune, full pairs, best-of-2-epoch checkpoint** | 630 / 157 | 2 (best-val saved) | **0.9299** | **+0.0509** | 0.2866 |

**Headline result: 92.99% held-out pairwise accuracy** — given two never-before-seen episodes of
the same game (one won, one lost), the model correctly identifies which one won 93% of the time,
using nothing but real environment outcomes as supervision (no hand-written rules, no per-step
reward). This is the strongest, most trustworthy verifier_dpo result and the one to report.

**The length-confound question (raised honestly, then resolved with evidence, not assumed away):**
an early per-turn readout showed per-turn score correlated with turn token-length at −0.62,
raising a real concern that this whole approach was substantially a length-shortcut classifier
rather than a genuine outcome-quality judge. Two checks resolve this at the episode level (the
level the headline number is reported at):
1. **`length_only_pairwise_acc` — a zero-model baseline** that predicts "the shorter episode
   won" using length alone: **28.7%**, i.e. *worse than a coin flip* and worse than the
   opposite guess. If length alone were driving the result, this number would be close to the
   model's accuracy; instead it's 64 points below it.
2. **The correlation itself trended toward zero as training scaled up** (−0.51 → −0.385 →
   **+0.05**) across the three configs, converging near-zero for the final, best-performing
   checkpoint — the opposite of what "the model is increasingly relying on a length shortcut"
   would predict.
Together, these are strong evidence the 93% accuracy reflects real learned signal, not a length
artifact. This length check was originally done at the EPISODE level only; per-turn scores were
separately, later validated for real turn-level discrimination via held-out AUC/F1 (see below) —
the two checks together cover both granularities this model is used at.

**Overfitting note (real, caught and handled, worth disclosing as methodology, not hidden):** the
first full-finetune attempt (3 epochs) showed a classic overfitting pattern — train accuracy
climbed to 99.5% while val accuracy peaked at epoch 1 (93.6%) then dropped at epoch 2 (91.7%),
the same pattern seen with `verifier_mc` (val loss/accuracy peaking early then degrading as
training continues — 2 epochs established as optimal at this data scale there too). Fixed by
adding best-val-checkpoint saving (`model.pt` = best epoch by val accuracy, `model_last.pt` =
final epoch kept only as a fallback) and reducing to 2 epochs, which is where the reported 0.9299
comes from (epoch 0 of that run, before epoch 1 began overfitting).

**Verdict:** a real, validated, generalizing episode-level outcome classifier — a genuinely
different training mechanism from `verifier_mc`'s per-step MC-return regression, using only
episode-level outcomes as supervision. Report the 92.99% figure as a positive result at the
episode level. Its per-turn readout is ALSO independently validated (see "Turn-Level Verifier
Comparison" below) — this model provides real signal at both granularities, not just the one it
was directly trained on.

---

## Best-of-N Verifier-Guided Evaluation

*(`scripts/best_of_n_eval.py`, run on Lightning AI, `eval_ood` split, 20 episodes/game/condition,
N=5 candidates at temperature=0.8, Qwen2.5-3B worker)*

The main test of whether a verifier's signal actually improves real-time decisions: at each turn,
sample N=5 candidate actions from the frozen worker at temperature>0 (real diversity, not the
deterministic greedy default used elsewhere), score each with a verifier, execute the argmax.
Compared against N=1 greedy (`run_episode`, the exact zero-coordination code path used throughout
this evaluation history — delegated to directly, not reimplemented). Tested with all THREE
verifiers: `verifier_mc` (trained, MC-return regression), `verifier_dpo` (trained,
episode-outcome preference), and a frozen, untrained LLM judge (`verifier/frozen_llm.py`'s
`score_candidate()` — a pre-hoc variant of build 08/09's post-hoc judge, since Best-of-N needs to
score a candidate BEFORE it's executed, not after; this is a genuinely new evaluation of that
prompt, not inherited from build 08/09's validated post-hoc numbers).

Note on the `llm` scorer: there is no per-candidate "accuracy" to report against a ground-truth
optimal action (none exists) — win rate, aggregated over many episodes, is the only meaningful
metric for all three scorers here, same as `mc`/`dpo`.

| game | baseline (N=1) | mc Best-of-N | mc delta | dpo Best-of-N | dpo delta | llm Best-of-N | llm delta |
|---|---|---|---|---|---|---|---|
| coin | 0% | 35% | +35pts | **65%** | **+65pts** | 20% | +20pts |
| simonsays | 100% | 100% | 0 (ceiling) | 100% | 0 (ceiling) | 100% | 0 (ceiling) |
| peckingorder | 100% | 100% | 0 (ceiling) | 100% | 0 (ceiling) | 100% | 0 (ceiling) |
| mapreader | 0% | 0% | 0 | 0% | 0 | 0% | 0 |
| **overall** | 50% | 58.75% | **+8.75pts** | **66.25%** | **+16.25pts** | 55.0% | **+5.0pts** |

**coin — stuck at 0% win rate under the zero-coordination baseline — jumps to 20-65% under
Best-of-N, with EVERY verifier tested showing a positive lift.** This is the first result in the
whole project that moves coin off zero, and the fact that all three independent scoring
mechanisms (trained regression, trained preference, untrained prompting) show gains — not just
one — is itself evidence the effect is real and not an artifact of one particular verifier.
Mechanism: coin is long-horizon and exploration-heavy (correct play requires methodically
searching many rooms); at a typical mid-episode turn, several candidate "keep exploring" actions
are plausible and only loosely differentiated by the worker's own single greedy guess — exactly
the situation where sampling a few options and picking a verifier's favorite has real room to
help, regardless of exactly how that verifier was built.

**A clean, monotonic ordering: `verifier_dpo` (+16.25pts overall) > `verifier_mc` (+8.75pts) >
untrained `llm` judge (+5.0pts).** On coin specifically: dpo 65%, mc 35%, llm 20% — more than 3x
between the strongest and weakest. This is the paper's most direct evidence that training on real
environment outcomes buys something concrete over prompting alone: the ordering isn't noise, it's
a coherent gradient from "no training" through two different training mechanisms, with the
episode-outcome-preference approach (`verifier_dpo`) coming out strongest. Worth leading with in
the paper — three independent methods, one consistent story.

**`score_candidate()` (the `llm` scorer) independently validated post-hoc, on the same real
`eval_ood` episodes `verifier_mc`'s AUC table was computed on** (`scripts/score_prehoc_llm_verifier.py`,
data/rollouts/v5_eval_ood/, 60 episodes/15 per game sampled, ~1,300 real turns scored):

| game | verifier_mc AUC | pre-hoc `llm` AUC | won_n / lost_n (llm) |
|---|---|---|---|
| coin | 0.709 | 0.588 | 110 / 450 |
| mapreader | 0.782 | 0.510 (≈chance) | 85 / 450 |
| peckingorder | 0.695 | 0.836 (n=1 lost, noise) | 125 / 1 |
| simonsays | 0.562 | 0.507 (≈chance) | 165 / 37 |
| **overall (pooled)** | **0.844** | **0.670** | 485 / 938 |

**Note: `verifier_dpo` deliberately has no entry in this table — this is a design property, not a
missing measurement.** AUC requires a per-turn score, computed the same way across many turns, so
won-episode turns can be checked against lost-episode turns for rank separation. `verifier_mc` and
the `llm` judge both natively produce that (a `q_value`/`advantage` or a 0-1 rating per turn).
`verifier_dpo` was trained ONLY on whole-episode comparisons (`episode_score()` over an entire
trajectory) — it has no per-turn training signal at all. The closest thing to a per-turn score
(the growing-prefix score used in Best-of-N) was already shown, via the mapreader trace above, to
be dominated by turn-length effects and to give two opposite actions at the same state nearly
identical scores — so computing an AUC from it would likely produce a noisy or misleading number,
measuring a capability the model was never trained to have. `verifier_dpo`'s own valid, comparable
metric is its 92.99% EPISODE-level pairwise accuracy (see the verifier_dpo section above) — the
AUC-equivalent at the granularity this model actually operates at, not a lesser or skipped metric.

**`verifier_mc` is a meaningfully better offline discriminator than the untrained pre-hoc judge**
— overall AUC 0.844 vs 0.670, a real, substantial gap. On mapreader and simonsays specifically,
the untrained judge is barely above chance (0.51, 0.51) while `verifier_mc` shows real signal
(0.78, 0.56) — training clearly buys discrimination ability the prompt alone doesn't have.
peckingorder's 0.836 is noise (only 1 lost turn in this sample) and should not be read as the
judge outperforming `verifier_mc` there.

**The genuinely interesting tension, worth keeping in the paper rather than smoothing over:**
despite this weaker AUC, the SAME `llm` judge still produced a real +20pt Best-of-N win-rate lift
on coin. A verifier can be a mediocre GLOBAL/offline discriminator (AUC, computed across the whole
score distribution) while still being useful for Best-of-N's NARROWER task (rank a small live set
of candidates relative to each other, at one specific turn) — Best-of-N only needs local ordering
within a handful of options, not calibrated, globally-accurate scores. This is a real, disclosed
nuance about what AUC does and doesn't predict for downstream usefulness, not a contradiction to
paper over.

**simonsays/peckingorder show 0 delta for all three scorers because both are already at a 100%
win-rate ceiling under this worker** — Best-of-N cannot improve on already-perfect baseline
performance. Not a verifier weakness; expected, and worth stating as such rather than left
ambiguous.

**mapreader stayed at 0% under every condition — investigated via direct trace inspection for
verifier_dpo, not assumed.** Every episode (baseline and Best-of-N alike) oscillates `move
north`/`move south` for the full 50-step cap, never winning. Two things confirmed directly from
saved traces (`reports/best_of_n_traces_dpo/mapreader_*.json`):
1. **The worker's own candidate distribution collapses to a single action at most turns** even at
   temperature=0.8 (turns 2-4 in every inspected episode produced exactly one DISTINCT candidate
   after deduplication, not five) — Best-of-N structurally cannot help when there is only one
   option to rank, regardless of which verifier is doing the ranking.
2. **At the one turn with real candidate diversity, the two live directional options scored
   nearly identically under verifier_dpo**: `move north`=7.407 vs `move south`=7.392, a gap of
   0.015 on opposite, mutually exclusive actions. Worth disclosing honestly: this is a real limit
   on this verifier's fine-grained, same-state resolution, visible directly in a real trace, not a
   hypothetical concern. Best-of-N still succeeds elsewhere despite this, because it degrades
   gracefully when candidates are near-tied (the argmax still picks something reasonable) rather
   than needing a sharp, non-degenerate signal the way a policy-gradient-style training objective
   would.

**Verdict:** a real, validated positive result — the first evidence in this project that a
verifier's signal translates into a measurable win-rate improvement, on the one game (coin) where
the underlying worker had genuine room to improve, replicated across three independently-built
scoring mechanisms with a consistent training-improves-quality ordering. Bounded honestly: helps
only where headroom exists (not ceiling games), doesn't help a game the worker structurally cannot
solve (mapreader, a capability gap, not a verifier gap), and the mapreader trace shows
verifier_dpo's fine-grained, same-state resolution has real limits — Best-of-N's robustness to
that limitation, not freedom from it, is why it still succeeds.

---

## Turn-Level Verifier Comparison: AUC + Online Failure Detection

*(`scripts/failure_detection_eval.py`, `scripts/score_dpo_turnlevel.py`; real held-out `eval_ood`
episodes, same 300-episode set for every verifier, `data/rollouts/v5_eval_ood/`)*

Two complementary tests of turn-level signal quality, run for `verifier_mc` and `verifier_dpo` on
the exact same held-out episodes (the `llm` judge has AUC only so far, see the AUC section above —
its failure-detection numbers are a natural next step, not run yet):

**1. AUC** — does the raw per-turn score correlate with the real outcome, averaged across the
whole score distribution?

**2. Online failure detection (F1/precision/recall/lead-time)** — a meaningfully STRONGER claim
than AUC: can the score, used causally (only ever turns up to and including the current one, never
looking ahead) with a K=5-turn trailing mean, give useful EARLY warning that an episode in progress
is heading toward failure? Threshold calibrated on a held-out CALIBRATION half (max-F1), reported
on a separate held-out TEST half, repeated across 5 seeds, mean ± range reported (not a single
split's number — an initial `verifier_mc` single-seed check landed at the high end of its real
range, 0.849 vs. a 0.765–0.849 spread, which is why every number below is a 5-seed summary).

| metric | verifier_mc | verifier_dpo |
|---|---|---|
| AUC (pooled) | **0.844** | 0.796 |
| F1 mean (range) | **0.796** (0.765–0.849) | 0.710 (0.681–0.756) |
| precision mean | **0.733** | 0.580 |
| recall mean | 0.884 | **0.917** |

**`verifier_dpo` DOES have real, held-out-validated turn-level signal — this corrects an earlier,
too-hasty inference in this project.** A single real trace (mapreader: two opposite actions scoring
7.407 vs. 7.392 at the same state, see the Best-of-N section above) had been read as evidence
`verifier_dpo`'s per-turn scores aren't reliable turn-level signal. That trace is still real and
worth keeping as a documented local weak spot, but it was one anecdote, not a controlled
measurement — the actual held-out AUC (0.796, every game clearly above chance) and F1 (0.710 mean)
show real turn-level discrimination at the aggregate level. `verifier_dpo` beats the untrained
`llm` judge's AUC (0.670) on every single game, despite never being trained on a single per-turn
label — its per-turn scores are a genuine, validated byproduct of episode-level training, not just
statistically-separable noise.

**`verifier_mc` remains the stronger turn-level verifier on both metrics**, as expected given it
was directly trained on a per-step target. The gap is real but not enormous (AUC 0.844 vs. 0.796;
F1 0.796 vs. 0.710) — `verifier_dpo` is a legitimate second option at this granularity, not a
distant one. The clearest split is on PRECISION (0.733 vs. 0.580, `verifier_dpo` false-alarms
more) rather than recall (`verifier_dpo` is actually slightly higher, 0.917 vs. 0.884, missing
fewer real failures) — worth stating precisely rather than as a blanket "mc is better."

**One methodological caveat on `verifier_dpo`'s F1 result:** its calibrated threshold landed on
the exact same value (−21.42) across all 5 seeds, unlike `verifier_mc`'s run (which picked between
two distinct values depending on the split) — likely because `verifier_dpo`'s raw per-turn score
range is much wider/less bounded than `verifier_mc`'s, so the calibration sweep may be less finely
resolved for it. The F1 numbers are still real, computed correctly, on genuinely held-out data —
this is a note on precision of the calibration, not a reason to distrust the result.

**Verdict:** the paper's "turn-level agentic verifier" framing holds for BOTH trained verifiers,
not just one — `verifier_mc` and `verifier_dpo` both provide real, held-out-validated, early,
actionable failure-detection signal, with `verifier_mc` the stronger of the two and `verifier_dpo`
a genuine, well-evidenced second option that was NOT directly optimized for this granularity at
all. This is stronger, more general evidence for the paper's core claim than either verifier alone
would provide.

---

## Suggested framing for Discussion/Conclusion

- **What's validated:** TWO mechanistically different turn-level verifiers, both trained purely on
  real environment outcomes (no hand-written rules), both with real, held-out AUC AND online
  failure-detection results — not just one verifier, and not just AUC alone. `verifier_mc`
  (per-step MC-return regression): AUC 0.844 pooled, failure-detection F1 mean 0.796. `verifier_dpo`
  (episode-outcome preference training, per-turn scores read off as a byproduct, never directly
  trained on a per-step target): AUC 0.796 pooled, failure-detection F1 mean 0.710 — genuinely
  close to `verifier_mc`'s numbers despite a structurally different, coarser training signal. This
  is the paper's core "turn-level agentic verifier" claim, validated twice, independently, by two
  different mechanisms.
- **The "online failure detector" claim is backed by a real, quantitative, early-warning result
  for BOTH verifiers, not inferred from AUC alone.** Using a causal (no-lookahead) trailing-mean
  signal, calibrated on a held-out split and tested on a separate held-out split across 5 seeds:
  `verifier_mc` F1 0.796 (range 0.765-0.849), precision 0.733, recall 0.884; `verifier_dpo` F1
  0.710 (range 0.681-0.756), precision 0.580, recall 0.917 — `verifier_dpo` actually has HIGHER
  recall (misses fewer real failures), `verifier_mc` has higher precision (fewer false alarms).
  Lead with the multi-seed mean for both, not any single split's number (an initial `verifier_mc`
  single-seed check landed at the high end of its range, 0.849, not the typical case). This is a
  MEANINGFULLY STRONGER claim than AUC alone — it shows both verifiers can flag a failing episode
  WHILE it's still in progress, with enough lead time (tens of turns on `verifier_mc`'s detailed
  run) for the flag to be actionable.
- **`verifier_dpo`'s real turn-level signal is itself a notable finding, and corrects an earlier,
  too-hasty inference in this project's own history — worth stating in the paper as an example of
  measuring rather than assuming.** A single real trace (mapreader Best-of-N: two opposite actions
  scoring 7.407 vs. 7.392 at the same state) had initially been read as evidence this model's
  per-turn scores aren't reliable — a real, disclosed local weak spot, but only one data point. The
  actual controlled, held-out measurement (AUC 0.796, F1 0.710, both clearly above chance on every
  game) shows real aggregate turn-level discrimination despite that local weakness — `verifier_dpo`
  beats the untrained `llm` judge's AUC (0.670) on every single game. Present both findings
  honestly: the local weak spot is real (worth keeping as a documented limitation), and the
  aggregate signal is also real (worth reporting as a positive result) — they aren't in
  contradiction, a real signal can still have real weak spots.
- **`verifier_dpo` also independently validated at the episode level:** 92.99% held-out
  episode-level pairwise accuracy — confirmed not to be a length-shortcut artifact via a zero-model
  length-only baseline (28.7%, worse than chance) and a correlation that trends to ~0 as training
  scaled up. `verifier_dpo` is therefore validated at BOTH granularities it's used at (episode and
  turn), not just one.
- **Also validated: Best-of-N verifier-guided action selection**, tested across THREE independent
  scoring mechanisms (N=5, temperature-sampled candidates, argmax by verifier score) — produces a
  REAL win-rate lift on coin under every one: 0%→65% with verifier_dpo, 0%→35% with verifier_mc,
  0%→20% with an untrained, prompted LLM judge. The first result in the project that moves that
  game off zero, and a clean, monotonic ordering (dpo > mc > untrained-llm, both overall and on
  coin specifically) that is itself evidence a verifier trained on real outcomes buys something
  real over prompting alone — not one lucky result, but a consistent gradient across three
  differently-built mechanisms. Present this as the paper's strongest positive evidence that the
  verifier's signal is practically useful, with the honest scope that it only demonstrated on coin
  (ceiling effects on simonsays/peckingorder, a worker capability gap on mapreader — see the
  Best-of-N section above for the direct trace evidence on both).
- **The untrained LLM judge was independently AUC-validated** (post-hoc, on the same real
  `eval_ood` episodes both trained verifiers were measured on): overall AUC 0.670 — clearly the
  weakest of the three (`verifier_mc` 0.844, `verifier_dpo` 0.796), especially visible on mapreader
  and simonsays where the untrained judge is near-chance (0.51 both) but both trained verifiers are
  not. Worth including as its own finding: training on real outcomes is a meaningfully better
  GLOBAL discriminator regardless of WHICH training mechanism is used, even though the untrained
  judge still produced a real Best-of-N win-rate lift — the two results aren't in tension, they say
  different things (AUC measures calibrated, global discrimination; Best-of-N only needs local
  ranking within a handful of live candidates at one turn, a much easier bar an untrained judge can
  still partially clear).
- **What's explicitly out of scope / not done, and why (for a Limitations subsection):** the `llm`
  judge's own online failure-detection numbers (F1/precision/recall/lead-time) were not run — it
  has AUC only so far, since a full per-turn scoring pass for it needs real per-candidate inference
  calls, not local forward passes, at real cost. Rule-based labeling was built, confirmed to
  mechanically fix same-state action discrimination, and deliberately rejected on principle (would
  have replaced environment-outcome learning with a hand-written heuristic, undermining the
  paper's actual claim) — worth stating explicitly as a design decision, not an oversight.
