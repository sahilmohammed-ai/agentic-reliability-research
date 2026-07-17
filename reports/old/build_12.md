# Build 12

**Date:** 2026-07-16

**Overview:** Re-test of coordinator v0 on ALFWorld after fixing the streak logic (build_11's root
cause) and the Thinker/Worker prompts. Full diagnosis, including a corrected reading of the
headline number, in `.info/BUILD_11_12_DIAGNOSIS.md`.

**Environment:** ALFWorld (TextWorld backend), train split, 50 episodes, Lightning AI.

**Agent setup:**
- Thinker + Worker: Qwen2.5-3B-Instruct, HF Transformers full precision, updated prompts
  (`agents/thinker.py` requirements-first plan format + few-shot examples; `agents/worker.py`
  examine/look discouragement + raw-output/parse_fallback logging).
- Coordinator: v0, streak decay fix (`max(0, streak-1)` instead of hard reset) + `backtrack` no
  longer clears `action_history`.
- `checkpoints/verifier_v1`, `Q_THRESHOLD=0.10` (unchanged).

**Metrics:**
- 50 episodes, 17 won (34.0% headline), 33 lost.
- **Sonnet-free subset (episodes with zero escalate turns), the fair frozen-3B-only comparison:
  8/50 won = 16.0%** — at or slightly below build_10's 20.4% baseline. `escalated_to` (a genuine
  model swap to `claude-sonnet-4-5`, not just a label) is set on 1,308/1,917 turns (68.2%), so the
  headline number is not usable evidence about the 3B pipeline on its own.
- Won episodes averaged 15.7 steps. All 33 losses hit the 50-step cap.
- Coordinator action histogram, 1,917 worker turns: continue 16.3%, retry 4.0%, replan 4.1%,
  backtrack 7.4%, **escalate 68.2%** (worse in raw share than build_11's 39.7%).
- Loop-breaking rate (ground truth: did a literal repetition loop actually break?): retry 17%->71%,
  replan 18%->77%, backtrack 25%->65% — confirms the streak/backtrack fix works, independent of
  the win-rate confound above.
- Worker: `parse_fallback` 0/1,917 turns. `examine` actions 39.2%->26.7%, but `look` rose to 19.7%
  (combined, arguably no better than build_10's baseline).

**Takeaway:** the streak-decay and backtrack-history fixes are real, mechanism-level progress.
Escalate's trigger frequency is now the top open problem — it's both unresolved on its own terms
and the direct cause of the win-rate confound. See `.info/BUILD_11_12_DIAGNOSIS.md` for the full
Sonnet-attribution analysis.
