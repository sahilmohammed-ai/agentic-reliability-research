# Build 11

**Date:** 2026-07-16

**Overview:** First multi-episode (n=50) test of coordinator v0 (`replan_mode=verifier`) on
ALFWorld, collected on Lightning AI. Full diagnosis in `.info/BUILD_11_12_DIAGNOSIS.md`.

**Environment:** ALFWorld (TextWorld backend), train split, 50 episodes.

**Agent setup:**
- Thinker + Worker: Qwen2.5-3B-Instruct, HF Transformers full precision.
- Coordinator: v0 (`rollout/coordinator.py`), live-verifier-driven, `checkpoints/verifier_v1`,
  `Q_THRESHOLD=0.10`.

**Metrics:**
- 50 episodes, 9 won (18.0%), 41 lost. Below build_10's mask-mode baseline (20.4%).
- Won episodes averaged 12.3 steps. All 41 losses hit the 50-step cap.
- Coordinator action histogram, 2,161 worker turns: continue 35.4%, retry 9.1%, replan 7.0%,
  backtrack 8.8%, **escalate 39.7%**.
- Escalate rate is nearly identical between won (47.75/100 turns) and lost (39.27/100 turns)
  episodes — it doesn't distinguish outcome at all.

**Root cause:** not the verifier (re-scored real turns live, generalizes correctly) — the
coordinator's `low_q_streak` has no decay and no ceiling, so it locks into `escalate` for the rest
of most episodes past ~10 steps, and `backtrack`'s context-clearing induces the thrashing it then
reacts to. See `.info/BUILD_11_12_DIAGNOSIS.md` for the full trace-level evidence.
