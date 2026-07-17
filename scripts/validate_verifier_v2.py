"""
validate a verifier checkpoint on the real held-out val split: percentile separation (won vs
lost turns), split further by data source (qwen vs sonnet) to check the checkpoint didn't
overfit toward one source at the expense of the other.

ALFWorld-specific (verifier v2, the alfworld + sonnet combined dataset). superseded as the active
track by the textworldexpress pivot (2026-07-17, see reports/twx/) after finding ALFWorld's
one-bit-per-episode terminal reward structurally caps how sharp turn-level separation can get, no
matter how the labels are constructed. kept as a working reference/comparison script, not deleted --
data/labeled/old/build_v2_combined still exists if this needs to be re-run.

usage:
    python -m scripts.validate_verifier_v2
"""

import json
import statistics

from verifier.dataset import load_examples
from verifier.infer import Verifier
from verifier.train import stratified_group_split

DATA_DIR = "data/labeled/old/build_v2_combined"
CHECKPOINT_DIR = "checkpoints/verifier_v2"


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "p10": s[int(n * 0.10)],
        "p25": s[int(n * 0.25)],
        "median": s[n // 2],
        "p75": s[int(n * 0.75)],
        "p90": s[int(n * 0.90)],
        "mean": statistics.mean(s),
    }


def main():
    examples = load_examples(DATA_DIR)
    train_idx, val_idx = stratified_group_split(examples, val_fraction=0.1)
    val_examples = [examples[i] for i in val_idx]
    print(f"held-out val examples: {len(val_examples)} across "
          f"{len(set(e['episode_id'] for e in val_examples))} episodes\n")

    verifier = Verifier(CHECKPOINT_DIR)

    # score every held-out example live (not reusing the stored label -- this tests the
    # checkpoint's actual live inference, the same path any consumer would use), re-reading each
    # episode's turns directly from its source file (load_examples() only kept flattened text,
    # not the individual task_goal/plan/obs_before/action fields needed for a fresh score() call)
    groups = {
        "all_won": [], "all_lost": [],
        "qwen_won": [], "qwen_lost": [],
        "sonnet_won": [], "sonnet_lost": [],
    }

    print("scoring held-out turns (this takes a while, one forward pass per turn)...")
    scored = 0
    for eid in sorted(set(e["episode_id"] for e in val_examples)):
        path = f"{DATA_DIR}/{eid}"
        with open(path) as f:
            traj = json.load(f)
        won = bool(traj["won"])
        source = "sonnet" if eid.startswith("sonnet_") else "qwen"
        turns = [t for t in traj["turns"] if t["role"] == "worker"]
        for t in turns:
            q, adv = verifier.score(traj["task_goal"], traj["plan"], t["obs_before"], t["action"])
            outcome_key = "won" if won else "lost"
            groups[f"all_{outcome_key}"].append(q)
            groups[f"{source}_{outcome_key}"].append(q)
            scored += 1

    print(f"scored {scored} turns\n")

    for name, values in groups.items():
        if not values:
            print(f"{name}: (no data)")
            continue
        p = percentiles(values)
        print(f"{name} (n={len(values)}): p10={p['p10']:.3f} median={p['median']:.3f} "
              f"p75={p['p75']:.3f} p90={p['p90']:.3f} mean={p['mean']:.3f}")


if __name__ == "__main__":
    main()
