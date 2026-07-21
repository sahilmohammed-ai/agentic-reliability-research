"""
label TextWorldExpress rollout trajectories with per-turn q_value and advantage using real
per-step env_reward, instead of rollout/label.py's ALFWorld-shaped flat won/lost Monte Carlo
decay.

rollout/label.py deliberately discards env_reward in favor of a terminal won/lost flag, because
ALFWorld's env_reward is 0 everywhere except 1.0 on the final winning step (a clean but
information-poor terminal signal) and using it directly would have been fine there, but
TextWorldExpress gives dense, real per-step reward (e.g. +0.125 on a correct placement, -0.125 on
an incorrect one) -- discarding it the same way throws away exactly the signal this project
pivoted environments to get access to (see .info/CLAUDE.md's "Pivot" section).

q_value = Monte Carlo return-to-go: the discounted sum of REAL future env_reward from this turn
to episode end (G_t = r_t + gamma*G_{t+1}). Built from real per-step reward at every future step,
not a terminal flag -- already a genuine upgrade over rollout/label.py's labels even without any
model involved.

advantage: NOT full GAE on this first pass. GAE needs a value estimate V(s) that the real reward
can disagree with (a TD residual delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)); if V(s) is set to the
EXACT Monte Carlo return-to-go, every residual is identically 0 by construction, since an exact MC
return trivially satisfies the Bellman equation -- confirmed directly by hand-deriving the
algebra and by testing on real build 03 data (an initial attempt at this produced all-zero
advantages, which is this mathematical fact showing up in practice, not an implementation bug to
route around). Using true GAE here would require a genuine, imperfect value estimate from a
trained model, which doesn't exist yet for TextWorldExpress -- the same bootstrap-circularity
problem this project's own prior research (.info/CLAUDE.md's "Prior-Art Code Findings") found real
adjacent implementations avoid via an iterative train-freeze-relabel scheme, not solved in one
pass. Instead: advantage = q_value(t) - q_value(t-1), a plain one-step value delta (same shape as
rollout/label.py's advantage, but now q_value is built from real per-step reward instead of flat
decay). True GAE is deferred to a later iteration, once a trained verifier exists to supply a real
V(s) estimate the reward can meaningfully disagree with.

usage:
    python -m rollout.label_twx --in data/rollouts/twx/03_qwen_stratified/coin \\
        --out data/labeled/twx/build_v3_coin
"""

import argparse
import glob
import json
import os

GAMMA = 0.99  # discount factor, matches rollout/label.py for consistency


def _mc_returns_to_go(rewards: list[float], gamma: float) -> list[float]:
    """discounted sum of real future reward from each turn to episode end (inclusive of the
    turn's own reward). computed backward in one pass: G_t = r_t + gamma * G_{t+1}, G_{T+1} = 0."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def label_trajectory(traj: dict, gamma: float = GAMMA) -> dict:
    """add q_value and advantage to every worker turn in one trajectory dict. q_value is the MC
    return-to-go from real env_reward; advantage is a one-step value delta (see module docstring
    for why this is NOT full GAE on this first pass)."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    rewards = [float(t["env_reward"]) for t in worker_turns]

    q_values = _mc_returns_to_go(rewards, gamma)

    prev_value = 0.0
    for turn, q in zip(worker_turns, q_values):
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
    print(f"{turns_labeled} worker turns now carry q_value (MC return-to-go, real reward) and advantage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, required=True, help="directory of rollout jsons")
    parser.add_argument("--out", dest="out_dir", type=str, required=True, help="directory for labeled jsons")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir, args.gamma)
