# TWX 01 — Capability Baseline

**Date:** 2026-07-18

**Overview:** Pivot from ALFWorld to TextWorldExpress, motivated by a structural finding: ALFWorld
gives essentially zero reward at every step except exactly 1.0 on the final winning action, which
caps how sharp a turn-level verifier's separation can get regardless of labeling scheme (confirmed
via direct audit of verifier v2: ~74% of training turns ended at q_value=0.0 exactly, even after a
real labeling improvement). TextWorldExpress gives genuine per-step reward on some of its games.
This report is a pure model-capability baseline (zero coordination, no verifier) before any
labeling/training work resumes on this environment. Full environment-selection research (why not
AgentPRM's own envs, Crafter, WebShop, ScienceWorld) is in `.info/CLAUDE.MD`.

**Environment:** TextWorldExpress, zero-coordination baseline (thinker plans once, worker acts
until done/cap, no replanning/masking/verifier). `--split eval_ood` (held-out test fold, never
used for training/tuning). 50-step cap.

**Agent setup:** Thinker + Worker, both `claude-sonnet-5`, Anthropic API. No coordinator, no
verifier.

**Run 1 — default game set (coin, simonsays, peckingorder), 50 episodes:**
- 46/50 won (92.0%)
- `coin`: 15/18 won (83%), avg 15.6 steps — real failures, real signal.
- `simonsays`: 31/32 won (97%), avg 8.1 steps — nearly solved, thin signal.
- `peckingorder`: 0 episodes sampled (random selection didn't draw it).

**Run 2 — previously-excluded harder games (cookingworld, twc, mapreader, arithmetic), 40
episodes.** These were excluded from the default set because a frozen Qwen2.5-3B worker solved
them near-0% (per TALES); re-tested with a frontier model instead:
- 14/40 won overall (35%).
- `arithmetic`: 10/10 won (100%), 50.0% of turns nonzero reward — trivially easy for Sonnet 5, no
  failure signal despite dense reward when it fires.
- `cookingworld`: 0/10 won (0%), 9.1% nonzero — genuinely hard, but real partial-credit reward
  exists mid-episode (not a broken/unsolvable substrate).
- `mapreader`: 4/13 won (31%), avg 36.6 steps, 2.5% nonzero — best win/loss balance of the four,
  but reward itself is sparse (mostly a single find/place event).
- `twc`: 0/7 won (0%), avg 50.0 steps (all hit the step cap), 42.6% nonzero — densest reward of
  everything tested. Initial trace inspection suggested a step-limit/completion-signaling issue
  (agent placing objects correctly but running out the clock) — re-tested below.

**`sorting` excluded regardless of model strength**: instant-death on a wrong move is a property
of the environment, not a capability gap (same reasoning that disqualified ScienceWorld's
"focus on X").

**Run 3 — `twc` re-test at double step limit (step_limit=100 vs default 50), 10 episodes,
`eval_ood`:** 0/10 won, all 10 hit the full 100-step cap. This rules out the step-limit
hypothesis cleanly. Full trace inspection (`data/rollouts/twx/01_twc_longer_steplimit/`) shows
why: the agent repeats the identical wrong action verbatim for the entire episode — e.g. `take
razor` / `put razor in dressing table` (−0.125, wrong location) / `take vase` / `put vase in
night stand` (−0.125, wrong location), looping this exact 4-action cycle from turn 1 through turn
100 with zero within-episode adaptation to the repeated negative signal. This is a genuine
task-comprehension/adaptation gap (the agent guesses a plausible-but-wrong "usual location" and
never updates), not a horizon problem — more steps just means more repetitions of the same
mistake.

**Final training mix decision:** drop `twc`. The dense reward is real, but the failure mode it
produces is narrow and repetitive (one wrong guess replayed verbatim) rather than diverse —
risks teaching the verifier "this exact repeated 4-action cycle is bad" rather than a
generalizable notion of progress. Final mix: `coin`, `simonsays`, `peckingorder`, `cookingworld`,
`mapreader` (`TRAINING_GAMES` in `envs/textworldexpress_env.py`). `peckingorder` still needs a
dedicated test (never sampled in runs 1–3 above). `arithmetic` stays excluded (100% win rate for
Sonnet 5, no failure signal despite dense reward).

**Notes:** `coin` remains the best-established single game (real signal, already validated at
scale). `mapreader` is a viable secondary option for task diversity, best win/loss balance (31%)
of the harder games. `arithmetic` and `simonsays` are both too easy for Sonnet 5 to contribute
meaningful failure signal; `cookingworld` is hard but not broken (real partial-credit reward
exists mid-episode), kept as a genuine-difficulty option.
