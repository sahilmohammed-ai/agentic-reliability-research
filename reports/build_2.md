# Build 2

**Date:** 2026-07-03

**Overview:** Performance test using Opus 4.8 as the frozen thinker/worker model, in place of Haiku. This run establishes the strong single-agent ceiling (baseline 2 in the evaluation plan), not a candidate for the frozen worker/thinker roles in the actual pipeline. Fixed the task_goal extraction bug from build_1 before this run.

**Environment:**
- ALFWorld (TextWorld backend)
- Train split

**Agent setup:**
- Thinker (Opus 4.8): generates a plan once at episode start
- Worker (Opus 4.8): picks one command per step from the admissible list
- No verifier agent involved
- Coordinator: fixed and hardcoded, same as build_1, runs thinker once then loops worker until done or 50 steps

**Metrics:**
- 50 episodes total
- 34 won, 16 lost (68% win rate)
- Won episodes averaged 12.3 steps
- Every lost episode hit the 50-step cap, none failed early, same pattern as build_1

**Notes:**
- Opus did not ceiling out at 100%, so this task set still has discriminating power at the frontier tier.
- Failure mode is still timeout-only across both builds so far. Worth checking if this is a property of the task set/step cap rather than the model.
- Next baseline candidate: a mid-tier model (Sonnet or a smaller open model) to see where the smaller-model-plus-our-approach comparison actually lives relative to this ceiling.
