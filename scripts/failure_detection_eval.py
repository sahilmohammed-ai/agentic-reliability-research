"""
online failure-detection evaluation: does the verifier's live per-turn score let you predict an
episode is going to fail BEFORE it actually ends, not just classify it correctly after the fact
(which is what AUC measures)? this is the actual "failure detector" capability the paper wants to
demonstrate -- a different, stronger claim than "the score correlates with outcome," proven by
showing early, actionable warning on real held-out episodes.

mechanism: replay each episode's real per-turn q_value trajectory (already scored, see
data/labeled/v5_eval_scores.json -- verifier_mc, no new compute needed) turn by turn. maintain a
TRAILING MEAN over the last K turns (raw single-turn q_value is too noisy to threshold directly --
confirmed via direct inspection: real lost-episode turns swing e.g. 0.35/0.02/0.38/-0.05 turn to
turn, while the trailing mean cleanly separates won vs lost, 0.47 vs 0.13 average final value).
flag the episode as PREDICTED-FAIL the first turn the trailing mean drops below a threshold. an
episode with no such turn is predicted-win.

methodology, to avoid a real bias (tuning and testing the threshold on the same data would
overstate performance): split episodes into a CALIBRATION half (choose the threshold that
maximizes F1 there) and a held-out TEST half (report precision/recall/F1/lead-time only on this
half, using the calibration-chosen threshold, never re-tuned). same spirit as this project's
train/val splits elsewhere.

lead time: for a correctly-predicted-fail episode (true lost, flagged), how many turns before the
episode's real end did the flag fire (total_steps - flag_turn). higher is better -- an
"early warning" that fires on the very last turn isn't useful for intervention; one that fires
with real turns to spare is. episodes with total_steps<=2 are excluded from lead-time averaging
(no meaningful "early" window to measure), but still count toward precision/recall.

usage:
    python -m scripts.failure_detection_eval --scores data/labeled/v5_eval_scores.json \\
        --score-field q_value --k 5 --out reports/failure_detection_mc.json
"""

import argparse
import json
import random


def trailing_mean(values: list[float], k: int) -> list[float]:
    """trailing_mean[i] = mean of values[max(0,i-k+1) : i+1] -- causal (only uses turns up to and
    including i), so this is a legitimate ONLINE signal, not a lookahead."""
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def first_flag_turn(trailing: list[float], threshold: float) -> int | None:
    """returns the 0-indexed turn where the trailing mean first drops below threshold, or None if
    it never does (episode predicted-win)."""
    for i, v in enumerate(trailing):
        if v < threshold:
            return i
    return None


def evaluate_at_threshold(
    episodes: list[dict], score_field: str, k: int, threshold: float,
) -> dict:
    tp = fp = tn = fn = 0
    lead_times = []
    for ep in episodes:
        values = [t[score_field] for t in ep["turns"]]
        if not values:
            continue
        trailing = trailing_mean(values, k)
        flag_turn = first_flag_turn(trailing, threshold)
        predicted_fail = flag_turn is not None
        actual_fail = not ep["won"]

        if predicted_fail and actual_fail:
            tp += 1
            if ep["total_steps"] > 2:
                lead_times.append(ep["total_steps"] - 1 - flag_turn)
        elif predicted_fail and not actual_fail:
            fp += 1
        elif not predicted_fail and actual_fail:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_lead_time": sum(lead_times) / len(lead_times) if lead_times else None,
        "n_lead_time_episodes": len(lead_times),
    }


def calibrate_threshold(
    episodes: list[dict], score_field: str, k: int, candidates: list[float],
) -> float:
    """sweeps candidate thresholds on the CALIBRATION set only, returns the one maximizing F1."""
    best_threshold, best_f1 = candidates[0], -1.0
    for thr in candidates:
        result = evaluate_at_threshold(episodes, score_field, k, thr)
        if result["f1"] > best_f1:
            best_f1, best_threshold = result["f1"], thr
    return best_threshold


def run(scores_path: str, score_field: str, k: int, out_path: str, seed: int = 42) -> None:
    with open(scores_path) as f:
        all_episodes = json.load(f)
    all_episodes = [ep for ep in all_episodes if ep["turns"]]

    # calibration/test split, stratified by game so both halves keep a representative game mix
    # (per-game won/lost counts are uneven, e.g. peckingorder 68 won/7 lost -- a flat random split
    # could leave the calibration half with too few lost episodes of some game to threshold on).
    rng = random.Random(seed)
    by_game: dict[str, list[dict]] = {}
    for ep in all_episodes:
        by_game.setdefault(ep["game"], []).append(ep)

    calibration, test = [], []
    for game, eps in by_game.items():
        eps = sorted(eps, key=lambda e: e["task_id"])
        rng.shuffle(eps)
        half = len(eps) // 2
        calibration.extend(eps[:half])
        test.extend(eps[half:])

    print(f"calibration: {len(calibration)} episodes, test: {len(test)} episodes")

    all_values = [t[score_field] for ep in all_episodes for t in ep["turns"]]
    lo, hi = min(all_values), max(all_values)
    candidates = [lo + (hi - lo) * i / 40 for i in range(41)]

    threshold = calibrate_threshold(calibration, score_field, k, candidates)
    print(f"calibrated threshold (max F1 on calibration set): {threshold:.4f}")

    calibration_result = evaluate_at_threshold(calibration, score_field, k, threshold)
    test_result = evaluate_at_threshold(test, score_field, k, threshold)

    print(f"\ncalibration-set performance at this threshold: {calibration_result}")
    print(f"\nHELD-OUT TEST-set performance (the real result): {test_result}")

    # per-game breakdown on the test set, using the same calibrated threshold
    per_game = {}
    for game, eps in by_game.items():
        game_test = [ep for ep in test if ep["game"] == game]
        if game_test:
            per_game[game] = evaluate_at_threshold(game_test, score_field, k, threshold)
    print("\nper-game test-set breakdown:")
    for game, r in per_game.items():
        print(f"  {game}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1']:.3f} mean_lead_time={r['mean_lead_time']}")

    results = {
        "score_field": score_field, "k": k, "threshold": threshold,
        "n_calibration": len(calibration), "n_test": len(test),
        "calibration_result": calibration_result,
        "test_result": test_result,
        "per_game_test": per_game,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results -> {out_path}")


def run_multi_seed(
    scores_path: str, score_field: str, k: int, out_path: str, seeds: list[int],
) -> None:
    """runs the full calibrate/test pipeline independently for each seed (a different
    calibration/test split each time) and reports the mean +/- range of test-set F1/precision/
    recall across seeds, not a single split's number -- added after confirming a real, honest
    result requires this: an initial single-seed run (F1=0.849) turned out to be the high end of a
    0.765-0.849 range across 5 seeds, not representative on its own."""
    all_results = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        with open(scores_path) as f:
            all_episodes = json.load(f)
        all_episodes = [ep for ep in all_episodes if ep["turns"]]

        rng = random.Random(seed)
        by_game: dict[str, list[dict]] = {}
        for ep in all_episodes:
            by_game.setdefault(ep["game"], []).append(ep)
        calibration, test = [], []
        for eps in by_game.values():
            eps = sorted(eps, key=lambda e: e["task_id"])
            rng.shuffle(eps)
            half = len(eps) // 2
            calibration.extend(eps[:half])
            test.extend(eps[half:])

        all_values = [t[score_field] for ep in all_episodes for t in ep["turns"]]
        lo, hi = min(all_values), max(all_values)
        candidates = [lo + (hi - lo) * i / 40 for i in range(41)]
        threshold = calibrate_threshold(calibration, score_field, k, candidates)
        test_result = evaluate_at_threshold(test, score_field, k, threshold)
        test_result["seed"] = seed
        test_result["threshold"] = threshold
        print(f"  threshold={threshold:.4f} test: precision={test_result['precision']:.3f} "
              f"recall={test_result['recall']:.3f} f1={test_result['f1']:.3f}")
        all_results.append(test_result)

    f1s = [r["f1"] for r in all_results]
    precisions = [r["precision"] for r in all_results]
    recalls = [r["recall"] for r in all_results]
    summary = {
        "seeds": seeds,
        "per_seed_test_results": all_results,
        "f1_mean": sum(f1s) / len(f1s), "f1_min": min(f1s), "f1_max": max(f1s),
        "precision_mean": sum(precisions) / len(precisions),
        "recall_mean": sum(recalls) / len(recalls),
    }
    print(f"\nACROSS {len(seeds)} SEEDS: f1 mean={summary['f1_mean']:.3f} "
          f"(range {summary['f1_min']:.3f}-{summary['f1_max']:.3f}), "
          f"precision mean={summary['precision_mean']:.3f}, recall mean={summary['recall_mean']:.3f}")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default="data/labeled/v5_eval_scores.json")
    parser.add_argument("--score-field", type=str, default="q_value",
                         help="which per-turn field to threshold (q_value or advantage)")
    parser.add_argument("--k", type=int, default=5, help="trailing-mean window size")
    parser.add_argument("--out", type=str, default="reports/failure_detection_mc.json")
    parser.add_argument("--single-seed", type=int, default=None,
                         help="run once with this seed only (prints per-game breakdown too). "
                              "default: run the multi-seed summary (5 seeds, no per-game detail).")
    parser.add_argument("--seeds", type=str, default="42,1,2,3,100",
                         help="comma-separated seeds for the multi-seed summary")
    args = parser.parse_args()
    if args.single_seed is not None:
        run(args.scores, args.score_field, args.k, args.out, seed=args.single_seed)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        run_multi_seed(args.scores, args.score_field, args.k, args.out, seeds)
