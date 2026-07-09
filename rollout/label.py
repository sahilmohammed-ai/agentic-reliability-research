"""
label rollout trajectories with per-turn q-values and advantages.

first-pass labeling: the only reward is the terminal win/loss signal, so the q-value for
each worker turn is the discounted return-to-go (gamma^(steps remaining) * terminal reward)
and the advantage is the change in value produced by the turn (value after minus value
before). once a trained verifier exists, these labels can be regenerated with proper td
bootstrapping the way agentprm does. for now this is pure monte carlo from the episode
outcome, no model needed.

usage:
    python -m rollout.label --in data/rollouts/build_5 --out data/labeled/build_5
"""

import argparse
import glob
import json
import os

# discount factor: how much each extra step to the goal reduces present value
GAMMA = 0.99


def label_trajectory(traj: dict, gamma: float = GAMMA) -> dict:
    """add q_value and advantage to every worker turn in one trajectory dict."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    n = len(worker_turns)
    # terminal reward comes from the last worker turn (1.0 on a win, 0.0 otherwise)
    terminal_reward = worker_turns[-1]["env_reward"] if n > 0 else 0.0

    # q-value at turn i = discounted return-to-go, value before the episode's first
    # action serves as the baseline for the first advantage
    prev_value = (gamma ** n) * terminal_reward
    for i, turn in enumerate(worker_turns):
        q = (gamma ** (n - 1 - i)) * terminal_reward
        turn["q_value"] = round(q, 6)
        turn["advantage"] = round(q - prev_value, 6)
        prev_value = q

    return traj


def label_directory(in_dir: str, out_dir: str, gamma: float = GAMMA) -> None:
    """label every trajectory json in in_dir and write results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise SystemExit(f"no trajectory files found in {in_dir}")

    won = 0
    turns_labeled = 0
    for path in files:
        with open(path) as f:
            traj = json.load(f)

        traj = label_trajectory(traj, gamma)
        won += int(traj["won"])
        turns_labeled += sum(1 for t in traj["turns"] if t["role"] == "worker")

        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(traj, f, indent=2)

    print(f"labeled {len(files)} trajectories ({won} won) -> {out_dir}/")
    print(f"{turns_labeled} worker turns now carry q_value and advantage")


if __name__ == "__main__":
    # parse cli arguments and label rollouts
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, required=True, help="directory of rollout jsons")
    parser.add_argument("--out", dest="out_dir", type=str, required=True, help="directory for labeled jsons")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir, args.gamma)
