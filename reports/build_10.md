# Build 10

**Date:** 2026-07-13

**Overview:** Large-scale rollout collection to produce verifier v1's training data. Two environments collected under identical config (mask mode, HF Qwen2.5-3B), so the labeled dataset covers two different action-vocabulary and admissible-command profiles rather than a single environment. ALFWorld collected locally, TextWorldExpress collected on Lightning AI.

**Environment:**
- ALFWorld (TextWorld backend), train split, 500 episodes
- TextWorldExpress (coin/simonsays/peckingorder games), train split, 500 episodes

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, HF Transformers full precision): plan at episode start, replans on stagnation
- Worker (Qwen2.5-3B-Instruct, HF Transformers full precision): picks one command per step from the admissible list
- No verifier agent involved (verifier did not exist yet at collection time)
- Coordinator: fixed, mask mode (stagnation-triggered replan plus single-turn masking of the most-repeated action), identical config across both environments

**Metrics:**
- ALFWorld: 500 episodes, 102 won (20.4%), 398 lost. Won averaged 14.7 steps. Every loss hit the 50-step cap.
- TextWorldExpress: 500 episodes, 377 won (75.4%), 123 lost. Won averaged 9.1 steps. Only 50 of 123 losses hit the step cap, the rest ended early.
- Combined: 1,000 episodes, 27,459 labeled worker turns (21,399 ALFWorld, 6,060 TextWorldExpress) after running rollout.label on both and merging into data/labeled/build_10_combined.

**Notes:**
- ALFWorld's win rate (20.4%) and all-losses-hit-cap pattern is consistent with every prior ALFWorld build. TextWorldExpress's much higher win rate (75.4%) and partial early-termination pattern reflects the easier task set (coin/simonsays/peckingorder, chosen because TALES shows a 3B model solves these at a real, high rate).
- This was a deliberate scope decision: verifier v1 trains on ALFWorld + TextWorldExpress only, not a third environment. WebShop was considered but its free-text search[...] action space does not fit the current admissible-command-list worker design, and building that harness redesign was explicitly deferred rather than blocking verifier training on it.
- Combining the two labeled sets required prefixing filenames (both environments independently number files train_0000 through train_0499), otherwise the second copy silently overwrote the first.
- This dataset directly supersedes an earlier, smaller ALFWorld-only labeled set that was used to train a first verifier checkpoint whose weights were never downloaded from the training Studio and are now considered lost/irrelevant. Build 10 is the first labeled dataset behind a verifier checkpoint that was actually retrieved and used.
