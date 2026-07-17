# Build 8

**Date:** 2026-07-10

**Overview:** Tested a hard intervention the worker cannot ignore: on a stagnation trigger, the single most-repeated recent action is masked out of the admissible set for that one turn (then restored), on top of the stagnation replan from build_7. Motivated by build_7's finding that replanning is a weak lever, it only suggests a new plan the worker often ignores, whereas masking forces a different choice. Same model, backend, and task set as build_7 for direct comparison.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, HF Transformers, full precision): plan at episode start, replans on stagnation (same as build_7)
- Worker (Qwen2.5-3B-Instruct, HF Transformers, full precision): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed, "mask" mode. On a stagnation trigger (last 6 worker actions have <=3 unique, 4-step cooldown) it both replans AND masks the single most-repeated recent action for that one turn. Masking drops only the single most-frequent action (not the whole cycle) to minimize the risk of blocking an action that was actually needed, and restores the full set the next turn.

**Metrics:**
- 100 episodes total
- 26 won, 74 lost (26% win rate) -- highest of any Qwen2.5-3B build so far
- Won episodes averaged 13.2 steps
- Every lost episode hit the 50-step cap, same as all prior builds
- 78/100 episodes triggered at least one intervention; 454 mask+replan events fired

**Transition analysis (the real result):**
Because aggregate win rate hides churn (a change can rescue some tasks while breaking others and net to zero), the meaningful metric is per-task transitions against a baseline, tasks are seeded identically across builds so train_N is the same game everywhere.

- **vs build_5 (no-replan baseline):** rescued 13 (lost->won), broke 5 (won->lost), net +8.
- **vs build_7 (stagnation replan, same HF backend, its direct predecessor):** rescued 5, broke 0, net +5.

The build_7 comparison is the clean one: adding masking on top of build_7's replanning rescued 5 more tasks and broke zero previously-won tasks. This directly addresses the concern that a new intervention might sacrifice easy wins to chase hard ones, masking did not.


