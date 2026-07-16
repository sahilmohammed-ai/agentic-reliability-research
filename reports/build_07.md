# Build 7

**Date:** 2026-07-10

**Overview:** Tested whether triggering the thinker's replan based on detected worker stagnation (last 6 actions show 3 or fewer unique actions, i.e. stuck in a short cycle) outperforms build_6's fixed 15-step replan interval. Also switched the thinker/worker backend from Ollama (quantized) to HF Transformers (full precision, bfloat16) for the same Qwen2.5-3B-Instruct model.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, HF Transformers, full precision): generates a plan at episode start, replans whenever stagnation is detected (last 6 worker actions have <=3 unique), with a 4-step cooldown after each replan so it does not fire every single step
- Worker (Qwen2.5-3B-Instruct, HF Transformers, full precision): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same structure as build_1 through build_6, but replan trigger is now stagnation-based instead of a fixed interval

**Metrics:**
- 100 episodes total
- 21 won, 79 lost (21% win rate)
- Won episodes averaged 12.0 steps
- Every lost episode hit the 50-step cap, same as all prior builds
- 78/100 episodes triggered at least one replan; 594 replans fired in total, averaging 7.6 per episode that replanned

**Notes:**
- Win rate (21%) is close to build_6's 18% and build_3's 22%, all in the same range as prior Qwen2.5-3B runs. Not a clear improvement over build_6's fixed-interval approach.
- **Confound:** this build also switched the model backend from Ollama (quantized) to HF Transformers (full precision), so this is not a clean single-variable test against build_6. Any difference (or lack of one) could be due to the replan trigger, the backend/precision change, or both. A true apples-to-apples comparison would need a build_6-equivalent run on the same HF backend, or this same stagnation-mode run on Ollama.
- The stagnation trigger fires much more often than the fixed interval did (594 replans across the run, averaging 7.6 per episode vs build_6's fixed max of ~3 per 50-step episode). Despite replanning far more frequently, win rate did not meaningfully improve, which weakens the original hypothesis that replan timing (rather than plan adherence) was the main bottleneck.
- Failure mode split: 45/79 losses are the oscillation pattern, 34/79 are the completion-judgment pattern, both still well represented.
- Consistent with build_6's finding: replanning, however it is triggered, does not appear to fix the underlying issue. The worker's adherence to a revised plan is still the likely bottleneck, not how often or when the thinker intervenes.
- Next: either isolate the backend variable properly (Ollama vs HF on the same replan mode) or move on from replan-triggering ablations, since two different trigger strategies now show the same ceiling.
