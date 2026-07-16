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

