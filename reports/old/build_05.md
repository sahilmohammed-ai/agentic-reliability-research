# Build 5

**Date:** 2026-07-06

**Overview:** First large-scale rollout collection intended as real labeled training data, not just a baseline comparison. Qwen2.5-3B was chosen as the frozen model after build_3/build_4 showed 7B gave no meaningful reliability gain over 3B. Same fixed coordinator as all prior builds.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, local via Ollama): generates a plan once at episode start
- Worker (Qwen2.5-3B-Instruct, local via Ollama): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same as build_1 through build_4, runs thinker once then loops worker until done or 50 steps

**Metrics:**
- 500 episodes total
- 75 won, 425 lost (15% win rate)
- Won episodes averaged 11.7 steps
- Every lost episode hit the 50-step cap, same as all prior builds

**Notes:**
- Win rate is lower than build_3's 22% on 50 episodes, likely due to greater task-type diversity at this larger sample size, including harder heat/cool/clean variants underrepresented in the smaller run.
- Failure mode split is close to even here: 223/425 losses are the location-oscillation loop, 202/425 show the completion-judgment pattern (repeated examining without acting, or swapping held objects without finishing). Both failure modes are well represented, which is useful for the verifier.


