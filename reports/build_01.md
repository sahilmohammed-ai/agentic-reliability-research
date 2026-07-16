# Build 1

**Date:** 2026-07-02

**Overview:** First working rollout pipeline. Frozen agents, heuristic coordinator, no learning yet. Goal was to confirm the environment produces clean trajectories with a real success/failure signal.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Haiku 4.5): generates a plan once at episode start
- Worker (Haiku 4.5): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, runs Thinker once then loops Worker until done or 50 steps

**Metrics:**
- 50 episodes total
- 15 won, 35 lost (30% win rate)
- Won episodes averaged 11.7 steps
- Every lost episode hit the 50-step cap, none failed early
