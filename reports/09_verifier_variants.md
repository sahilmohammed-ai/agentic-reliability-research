# TWX 09 — Verifier Variants

**Date:** 2026-07-21

**Change:** tested two alternatives to build 08's frozen verifier (Qwen2.5-3B, reasoning-required
prompt): a no-reasoning (bare-number) prompt, and a Qwen2.5-1.5B backbone. Both offline, scored
against build 03's existing baseline.

**Won/lost mean score, reasoning vs. no-reasoning (Qwen2.5-3B):**

| game | reasoning won/lost | no-reasoning won/lost |
|---|---|---|
| coin | 0.403 / 0.272 | 0.097 / 0.006 |
| peckingorder | 0.777 / 0.360 | 0.306 / 0.200 |
| mapreader | 0.432 / 0.596 | 0.153 / 0.078 |
| **overall** | **0.583 / 0.431** | **0.209 / 0.069** |

**Insights:**

- No-reasoning compresses scores toward 0 (overall mean 0.209 vs 0.583) but keeps a similar
  won/lost gap — still separates outcomes relatively, just on a narrower, less interpretable scale.
- No-reasoning is ~25% faster (~15s vs ~20s/call) and actually fixes `peckingorder`'s inverted
  reward correlation, but loses resolution on `cookingworld`/`mapreader`.
- **Qwen2.5-1.5B backbone (smoke test only, no full run):** faster, but failed the exact case that
  caught build 08's original bug — self-contradictory reasoning scored a literal, correct
  instruction-match as 0.0, deterministic across repeats. A plain fixed-goal case scored fine, so
  the failure is scoped to reactive/instruction-matching judgments specifically. Not run further.

**Verdict:** keep Qwen2.5-3B with the reasoning-required prompt as the default. Neither
alternative is a clean win, and interpretable absolute scores matter if this becomes a coordinator
threshold.
