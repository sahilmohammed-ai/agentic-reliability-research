# Build 4

**Date:** 2026-07-05

**Overview:** Tested whether a larger open-source model fixes the completion-judgment failures seen with Qwen2.5-3B in build_3. Same setup, same task set, only the model size changed.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-7B-Instruct, local via Ollama): generates a plan once at episode start
- Worker (Qwen2.5-7B-Instruct, local via Ollama): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same as build_1, build_2, build_3, runs thinker once then loops worker until done or 50 steps

**Metrics:**
- 50 episodes total
- 12 won, 38 lost (24% win rate)
- Won episodes averaged 7.7 steps, faster than the 3B model's 13.4
- Every lost episode hit the 50-step cap, same as all prior builds

**Notes:**
- Win rate barely improved over 3B (24% vs 22%), essentially no reliability gain from the larger model.
- Won episodes are noticeably more efficient (7.7 avg steps vs 13.4), so when 7B succeeds it succeeds faster and more directly, but it does not succeed more often.
- Failure mode split got worse, not better: only 11/38 losses are the location-oscillation loop, the remaining 27/38 are the completion-judgment pattern from build_3 (examining objects repeatedly without acting, or swapping which object it holds without finishing the task). Scaling 3B to 7B did not fix this, and if anything the larger model shows it more.
- Current standings: Opus 4.8 (68%) > Haiku 4.5 (30%) > Qwen2.5-7B (24%) > Qwen2.5-3B (22%).
- Suggests the completion-judgment issue is not simply a capacity/size problem within the Qwen2.5 family at this scale. May need a prompting fix (e.g. explicit "check if task is complete" step) rather than a bigger model, or may be a good target failure mode for the verifier to learn to detect.
