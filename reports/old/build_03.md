# Build 3

**Date:** 2026-07-04

**Overview:** First open-source candidate for the actual frozen thinker/worker roles, Qwen2.5-3B-Instruct running locally via Ollama. This is the model AgentPRM validates on, so it is the natural starting point before committing to a frozen model for the labeled dataset. Also added multi-backend support (agents/llm.py) so thinker.py and worker.py can call either Anthropic or Ollama models through the same model parameter.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Qwen2.5-3B-Instruct, local via Ollama): generates a plan once at episode start
- Worker (Qwen2.5-3B-Instruct, local via Ollama): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same as build_1 and build_2, runs thinker once then loops worker until done or 50 steps

**Metrics:**
- 50 episodes total
- 11 won, 39 lost (22% win rate)
- Won episodes averaged 13.4 steps
- Every lost episode hit the 50-step cap, same as build_1 and build_2

**Notes:**
- Lowest win rate of the three baselines so far (Haiku 30%, Opus 68%, Qwen2.5-3B 22%).
- Failure mode is different from build_1/build_2. Only about half of the losses (18/39) are the same location-oscillation loop seen with Haiku and Opus. The other half show a new pattern: repeatedly examining the same object/location without acting, or swapping which object it is holding back and forth without ever finishing the task (e.g. moving cellphone 1 onto a table, then picking up cellphone 2, moving it, picking cellphone 1 back up, repeating). This looks like weaker judgment about task-completion conditions, not just a memory/location bug.
- Runs fast and fully locally, no API cost, useful if this becomes the frozen model for large-scale rollout collection.