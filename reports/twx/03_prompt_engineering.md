# TWX 03 — Prompt Engineering

**Date:** 2026-07-19

**Change:** `agents/thinker.py`'s system prompt was 100% ALFWorld-specific, causing bad plans on
TextWorldExpress's reactive games (e.g. `simonsays`, where the correct action changes every turn).
Rewrote it to classify each task as fixed-goal, reactive/instruction-following, or
discover-then-act (e.g. `cookingworld`'s hidden recipe), and plan accordingly.

**Metrics (Qwen2.5-3B, agentic loop, new prompt):**

| game | won | win rate | avg steps | nonzero-reward turns |
|---|---|---|---|---|
| coin | 13/20 | 65% | 27.6 | 2.4% |
| simonsays | 20/20 | 100% | 5.0 | 100.0% |
| peckingorder | 18/20 | 90% | 8.0 | 50.6% |
| cookingworld | 0/20 | 0% | 27.9 | 5.2% |
| mapreader | 7/20 | 35% | 37.2 | 2.7% |

**3-way comparison:**

| game | single-agent | loop, old prompt | loop, new prompt |
|---|---|---|---|
| coin | 60% | 55% | 65% |
| simonsays | 100% | 55% | 100% |
| peckingorder | 40% | 45% | 90% |
| cookingworld | 0% | 0% | 0% |
| mapreader | 30% | 40% | 35% |

**Insights:**

- `simonsays` and `peckingorder` (both reactive-classified) jumped sharply, closing or exceeding
  the single-agent gap — confirms the old loop's failure was prompt quality, not planning itself.
- `cookingworld` stayed at 0% — a genuine capability ceiling, not a prompt problem.
- Prompt fixes solve prompt-mismatch failures but not capability gaps; these need different fixes.
