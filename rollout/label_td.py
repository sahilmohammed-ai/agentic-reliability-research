"""
TD/GAE relabeling for TextWorldExpress trajectories, using a TRAINED verifier checkpoint as the
value function V(s). This is the second-iteration labeler in the train-freeze-relabel scheme:
rollout/label_twx.py produces first-pass Monte-Carlo labels (q_value = discounted real reward-
to-go), a first verifier is trained on those, then THIS script uses that frozen verifier's own
per-turn value predictions as V(s) to compute genuine TD residuals and GAE advantages.

Why this needs a trained model and label_twx.py's plain MC could not do it: a TD residual is
delta_t = r_t + gamma*V(s_{t+1}) - V(s_t). If V(s) is the EXACT Monte-Carlo return-to-go, every
residual is identically zero by construction (an exact MC return trivially satisfies the Bellman
equation) -- confirmed by direct derivation earlier in this project. GAE only produces meaningful,
action-dependent advantages when V(s) is an APPROXIMATION the real reward can disagree with. A
trained verifier is exactly such an imperfect estimate, so its residuals are nonzero and reflect
whether each action did better or worse than the model expected -- action-dependent signal, unlike
label_twx.py's position-dependent MC decay.

value semantics: the trained verifier's q_value output is treated as V(s_t), the value estimate
for the state at turn t. (The checkpoint's two heads are q_value and advantage; we use the q_value
head as the state-value estimate for bootstrapping.) V(s_{t+1}) for the final turn is 0 (no state
after the episode ends).

new labels written per worker turn:
  q_value   = the trained verifier's own V(s_t) prediction (the target the NEXT iteration regresses
              toward -- carrying the model's value estimate forward, standard for iterative PRM).
  advantage = GAE(lambda) over the TD residuals delta_t = r_t + gamma*V(s_{t+1}) - V(s_t).

usage:
    python -m rollout.label_td --in data/rollouts/<build>/coin \\
        --out data/labeled/<build>_td/coin --checkpoint checkpoints/verifier_v1
"""

import argparse
import glob
import json
import os

from verifier.infer import Verifier

GAMMA = 0.99
LAMBDA = 0.95


def _gae(rewards: list[float], values: list[float], gamma: float, lam: float) -> list[float]:
    """GAE(lambda) over TD residuals. values[t] = V(s_t) (from the trained verifier). the
    residual uses V(s_{t+1}) as the bootstrap; V beyond the last turn is 0."""
    n = len(rewards)
    advantages = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages


def label_trajectory(traj: dict, verifier: Verifier, gamma: float = GAMMA, lam: float = LAMBDA) -> dict:
    """relabel every worker turn using the trained verifier as V(s). q_value becomes the
    verifier's own V(s_t) prediction; advantage becomes GAE over the TD residuals."""
    worker_turns = [t for t in traj["turns"] if t["role"] == "worker"]
    rewards = [float(t["env_reward"]) for t in worker_turns]

    # V(s_t): the trained verifier's q_value prediction for each state. one forward pass per turn.
    plan = traj.get("plan", "")
    values = [
        verifier.score(traj["task_goal"], plan, t["obs_before"], t["action"])[0]
        for t in worker_turns
    ]

    advantages = _gae(rewards, values, gamma, lam)

    for turn, v, adv in zip(worker_turns, values, advantages):
        turn["q_value"] = round(v, 6)
        turn["advantage"] = round(adv, 6)

    return traj


def label_directory(
    in_dir: str, out_dir: str, checkpoint: str, gamma: float = GAMMA, lam: float = LAMBDA
) -> None:
    """relabel every trajectory json in in_dir and write results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise SystemExit(f"no trajectory files found in {in_dir}")

    # v3+ TextWorldExpress checkpoints train with unbounded q_value (real, sometimes-negative
    # reward), so load with bound_q_value=False -- otherwise every V(s) gets sigmoid-squashed.
    verifier = Verifier(checkpoint, bound_q_value=False)

    won = 0
    turns_labeled = 0
    for path in files:
        with open(path) as f:
            traj = json.load(f)

        traj = label_trajectory(traj, verifier, gamma, lam)
        won += int(traj["won"])
        turns_labeled += sum(1 for t in traj["turns"] if t["role"] == "worker")

        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(traj, f, indent=2)

    print(f"relabeled {len(files)} trajectories ({won} won) -> {out_dir}/")
    print(f"{turns_labeled} worker turns now carry TD/GAE q_value (V(s) from {checkpoint}) and advantage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, required=True, help="directory of rollout jsons")
    parser.add_argument("--out", dest="out_dir", type=str, required=True, help="directory for relabeled jsons")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained verifier checkpoint for V(s)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    parser.add_argument("--lam", type=float, default=LAMBDA, help="GAE lambda")
    args = parser.parse_args()
    label_directory(args.in_dir, args.out_dir, args.checkpoint, args.gamma, args.lam)
