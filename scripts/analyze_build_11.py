"""
diagnose a replan_mode=verifier (coordinator v0) rollout collection: win rate, action histogram,
and whether the coordinator's interventions actually break literal repetition loops. built for
build_11 (50-episode alfworld, 9/50 won, at or below build_10_alfworld's mask-mode baseline of
102/500 = 20.4%); reused as-is (via --dir) for re-tests after coordinator fixes, e.g. build_12.

sections:
  1. threshold miscalibration: is q_threshold=0.10 trigger-happy on episodes that were actually
     fine, or too lax to catch real trouble? -> action histogram split by episode outcome.
  2. intervention design: do episodes with heavy backtrack/escalate activity still lose anyway
     (verifier right, intervention doesn't recover), or do won episodes also carry a lot of
     intervention (interventions are just noise, not harmful, not helpful)?
  3. escalate specifically: does the one-turn frontier-model swap ever precede a win, or does it
     only show up in episodes that lose regardless?
  4. loop-breaking: of the turns where an action fired WHILE the worker was in a literal,
     ground-truth repetition loop (same check as rollout/coordinator.py's _is_repeating()), does
     the action sequence actually diversify in the following turns, or does the same loop persist?
     this is the ground-truth check, independent of what the verifier's q_value says -- a q_value
     improving after an action fires does not by itself mean the loop broke (confirmed on build_11:
     q_value improved after backtrack in 59% of triggers, but the literal loop only broke 25% of
     the time even for backtrack, the best performer of the four actions).

usage:
    python -m scripts.analyze_build_11 --dir data/rollouts/build_11
    python -m scripts.analyze_build_11 --dir data/rollouts/build_12   # re-test after a fix
"""

import argparse
import glob
import json
from collections import Counter


ACTIONS = ["continue", "retry", "replan", "backtrack", "escalate"]

# same repetition definition as rollout/coordinator.py's _is_repeating()
STAGNATION_WINDOW = 6
STAGNATION_MAX_UNIQUE = 3


def load_trajectories(in_dir: str) -> list[dict]:
    files = sorted(glob.glob(f"{in_dir}/*.json"))
    if not files:
        raise SystemExit(f"no trajectory files found in {in_dir}")
    return [json.load(open(f)) for f in files]


def worker_turns(traj: dict) -> list[dict]:
    return [t for t in traj["turns"] if t["role"] == "worker"]


def action_histogram(turns: list[dict]) -> Counter:
    return Counter(t["metadata"].get("coordinator_action", "continue") for t in turns)


def is_repeating(action_hist: list[str]) -> bool:
    if len(action_hist) < STAGNATION_WINDOW:
        return False
    recent = action_hist[-STAGNATION_WINDOW:]
    return len(set(recent)) <= STAGNATION_MAX_UNIQUE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/rollouts/build_11")
    args = parser.parse_args()

    trajs = load_trajectories(args.dir)
    n = len(trajs)
    won = [t for t in trajs if t["won"]]
    lost = [t for t in trajs if not t["won"]]

    print(f"loaded {n} episodes: {len(won)} won ({100*len(won)/n:.1f}%), {len(lost)} lost\n")

    # 1. overall action histogram, all turns pooled
    all_turns = [t for traj in trajs for t in worker_turns(traj)]
    overall = action_histogram(all_turns)
    print("overall coordinator action histogram (all turns, all episodes):")
    for a in ACTIONS:
        c = overall.get(a, 0)
        print(f"  {a:<10} {c:>5}  ({100*c/max(len(all_turns),1):.1f}%)")
    print(f"  total worker turns: {len(all_turns)}\n")

    # 2. action histogram split by episode outcome -- diagnoses threshold miscalibration (case 1)
    won_turns = [t for traj in won for t in worker_turns(traj)]
    lost_turns = [t for traj in lost for t in worker_turns(traj)]
    won_hist = action_histogram(won_turns)
    lost_hist = action_histogram(lost_turns)

    print("action histogram split by episode outcome (rate per 100 turns, so group sizes are comparable):")
    print(f"  {'action':<10} {'won (n='+str(len(won_turns))+')':<20} {'lost (n='+str(len(lost_turns))+')':<20}")
    for a in ACTIONS:
        won_rate = 100 * won_hist.get(a, 0) / max(len(won_turns), 1)
        lost_rate = 100 * lost_hist.get(a, 0) / max(len(lost_turns), 1)
        print(f"  {a:<10} {won_rate:>6.2f} per 100 turns    {lost_rate:>6.2f} per 100 turns")
    print()

    # 3. per-episode intervention density vs outcome -- do episodes that escalate/backtrack a lot
    #    still lose anyway (verifier right, intervention doesn't recover), or does it not correlate?
    print("per-episode escalate/backtrack counts (won vs lost), first 50 chars of task goal:")
    for traj in trajs:
        turns = worker_turns(traj)
        hist = action_histogram(turns)
        esc, back = hist.get("escalate", 0), hist.get("backtrack", 0)
        if esc or back:
            outcome = "WON " if traj["won"] else "lost"
            goal = traj["task_goal"][:50]
            print(f"  [{outcome}] steps={traj['total_steps']:<3} escalate={esc} backtrack={back}  {goal}")
    print()

    # 4. escalate specifically: does it ever precede eventual recovery within the same episode?
    #    check the q_value trend for a few turns after each escalate turn.
    print("q_value trend in the 3 turns after each escalate action:")
    escalate_count = 0
    for traj in trajs:
        turns = worker_turns(traj)
        for i, t in enumerate(turns):
            if t["metadata"].get("coordinator_action") == "escalate":
                escalate_count += 1
                after = turns[i + 1:i + 4]
                trend = [round(u["metadata"].get("q_value", float("nan")), 3) for u in after]
                outcome = "WON" if traj["won"] else "lost"
                print(f"  episode {traj['task_id']} ({outcome}) step {t['step']}: q after escalate = {trend}")
    if escalate_count == 0:
        print("  (no escalate actions occurred in this dataset)")
    print()

    # 5. loop-breaking: ground-truth check, independent of q_value. of the turns where an action
    #    fired WHILE the worker was literally stuck repeating, does the action sequence actually
    #    diversify in the next 6 turns, or is it still the same repeated set?
    print("loop-breaking rate (ground truth: was the worker literally repeating when the action fired?):")
    loop_results = {a: [] for a in ("retry", "replan", "backtrack", "escalate")}
    for traj in trajs:
        turns = worker_turns(traj)
        action_hist = []
        prev_coord_action = "continue"
        for i, t in enumerate(turns):
            coord_action = t["metadata"].get("coordinator_action", "continue")
            action_hist.append(t["action"])
            if (
                coord_action != "continue"
                and coord_action != prev_coord_action
                and coord_action in loop_results
                and is_repeating(action_hist)
                and i + 6 <= len(turns) - 1
            ):
                pre_loop_set = set(action_hist[-6:])
                future_actions = [turns[j]["action"] for j in range(i + 1, i + 7)]
                broke_loop = not is_repeating(action_hist + future_actions)
                still_same_loop = set(future_actions) <= pre_loop_set and len(set(future_actions)) <= 3
                loop_results[coord_action].append((broke_loop, still_same_loop))
            prev_coord_action = coord_action

    for action, records in loop_results.items():
        if not records:
            print(f"  {action:<10} n=0 (no loop-trigger cases with 6 future turns available)")
            continue
        broke = sum(1 for r in records if r[0])
        same = sum(1 for r in records if r[1])
        print(
            f"  {action:<10} n={len(records):<4} loop_broke={broke}/{len(records)} "
            f"({100*broke/len(records):.0f}%)  still_same_loop={same}/{len(records)} "
            f"({100*same/len(records):.0f}%)"
        )
    print()

    # 6. step-count comparison: does the coordinator burn extra steps without reward (cost of
    #    intervention), independent of whether it ultimately wins?
    won_steps = [t["total_steps"] for t in won]
    lost_steps = [t["total_steps"] for t in lost]
    if won_steps:
        print(f"won episodes: avg {sum(won_steps)/len(won_steps):.1f} steps")
    if lost_steps:
        print(f"lost episodes: avg {sum(lost_steps)/len(lost_steps):.1f} steps "
              f"({sum(1 for s in lost_steps if s >= 50)}/{len(lost_steps)} hit the 50-step cap)")


if __name__ == "__main__":
    main()
