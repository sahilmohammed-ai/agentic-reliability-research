# Build 6

**Date:** 2026-07-07

**Overview:** Tested whether adding mid-episode replanning to the fixed coordinator reduces the oscillation and completion-judgment failures seen in builds 1-5. The thinker now re-invokes every 15 worker steps, seeing the full action history, and can revise the plan. Same model and task set as build_5 for direct comparison.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, local via Ollama): generates a plan at episode start, then replans every 15 worker steps given the full action history and current scene
- Worker (Qwen2.5-3B-Instruct, local via Ollama): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same as prior builds, with the addition of the 15-step replan trigger

**Metrics:**
- 100 episodes total
- 18 won, 82 lost (18% win rate)
- Won episodes averaged 10.4 steps
- Every lost episode hit the 50-step cap, same as all prior builds
- 87/100 episodes triggered at least one replan; 13 finished or failed before step 15

**Notes:**
- Modest improvement over build_5 (15% on 500 episodes), but sample sizes are small enough (100 vs 500) that this is not a strong signal on its own. Build_3's 50-episode run alone showed 22%, so run-to-run variance at this scale is real.
- Replanning fires reliably (87% of episodes) but does not clearly fix the underlying failure. Of the 82 losses, 54 are still the completion-judgment pattern and 28 are still the location-oscillation loop, both present after at least one replan occurred.
