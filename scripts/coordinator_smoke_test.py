"""
smoke test for coordinator v0 (rollout/coordinator.py) with the repetition-gap fix in place.

runs one real alfworld episode through rollout.runner.run_episode with replan_mode="verifier",
using the hf backend (worker/thinker model "hf:Qwen/Qwen2.5-3B-Instruct") and the trained
verifier checkpoint at checkpoints/verifier_v1. prints the per-turn q_value, advantage, and
coordinator_action so the repetition-gap fix (_is_repeating() as a second trigger alongside
the q-value streak) can be visually confirmed: an escalating coordinator_action should appear
even during a run of literally-repeated actions where q_value stays stable and above
Q_THRESHOLD, not just during a low-q streak.

usage:
    python -m scripts.coordinator_smoke_test
"""

from envs.alfworld_env import AlfWorldEnv
from rollout.runner import run_episode
from verifier.infer import Verifier

MODEL = "hf:Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT_DIR = "checkpoints/verifier_v1"


def main():
    verifier = Verifier(CHECKPOINT_DIR)
    env = AlfWorldEnv("configs/alfworld_base.yaml", split="train")

    traj = run_episode(env, model=MODEL, replan_mode="verifier", verifier=verifier)
    env.close()

    print(f"\ntask: {traj.task_goal}")
    print(f"won: {traj.won}, total_steps: {traj.total_steps}\n")

    streak = 0
    for turn in traj.turns:
        if turn.role == "thinker":
            trigger = turn.metadata.get("trigger", "?")
            print(f"[step {turn.step}] THINKER ({turn.metadata.get('type', 'plan')}, trigger={trigger})")
            continue

        q = turn.metadata.get("q_value")
        adv = turn.metadata.get("advantage")
        action = turn.metadata.get("coordinator_action", "-")
        streak = streak + 1 if action != "continue" else 0
        q_str = f"{q:.3f}" if q is not None else "  -  "
        print(
            f"[step {turn.step:>2}] action={turn.action!r:<35} q={q_str} coordinator={action:<10} "
            f"reward={turn.env_reward}"
        )

    escalations = [t for t in traj.turns if t.metadata.get("coordinator_action") == "escalate"]
    backtracks = [t for t in traj.turns if t.metadata.get("coordinator_action") == "backtrack"]
    print(f"\nescalate turns: {len(escalations)}, backtrack turns: {len(backtracks)}")


if __name__ == "__main__":
    main()
