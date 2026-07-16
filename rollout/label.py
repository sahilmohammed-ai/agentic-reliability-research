"""
label rollout trajectories with per-turn q-values and advantages.

first-pass labeling: the only reward is the terminal win/loss signal, so the q-value for
each worker turn is the discounted return-to-go (gamma^(steps remaining) * terminal reward)
and the advantage is the change in value produced by the turn (value after minus value
before). once a trained verifier exists, these labels can be regenerated with proper td
bootstrapping the way agentprm does. for now this is pure monte carlo from the episode
outcome, no model needed.

terminal reward is derived from traj["won"] (1.0 if won, 0.0 if lost), NOT from the last
worker turn's raw env_reward. this matters across environments: alfworld's env_reward is 0
everywhere except a clean 1.0 on the winning step, but textworldexpress gives dense,
cumulative rewards (e.g. +0.25 per correct sub-step, negative on wrong ones), so its last
step's raw value can be anything (0.2, 0.25, -1.8, ...) depending on when the episode ended,
not a clean terminal signal. using won/lost directly keeps q-value scale identical across
every environment by construction, instead of silently training on inconsistent label scales
per environment (caught by inspecting build_10_textworldexpress: only 120 of 377 wins had
terminal env_reward == 1.0, the rest were 0.2/0.25; some losses had terminal env_reward as
extreme as -1.8).

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
    # terminal reward from the won/lost flag, not the last turn's raw env_reward (see
    # module docstring for why: raw env_reward isn't a consistent terminal signal across
    # environments with dense/cumulative rewards like textworldexpress).
    terminal_reward = 1.0 if traj.get("won") else 0.0

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
