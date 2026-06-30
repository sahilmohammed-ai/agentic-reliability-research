from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    step: int
    role: str           # "thinker" | "worker" | "verifier_role"
    obs_before: str
    action: str         # for worker: the env command chosen; for others: the text output
    obs_after: str      # empty string for non-env-stepping turns
    env_reward: float   # non-zero only for worker turns that call env.step()
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "role": self.role,
            "obs_before": self.obs_before,
            "action": self.action,
            "obs_after": self.obs_after,
            "env_reward": self.env_reward,
            "done": self.done,
            "metadata": self.metadata,
        }


@dataclass
class Trajectory:
    task_id: str
    task_goal: str
    plan: str           # thinker's high-level plan, generated at episode start
    turns: list[Turn]
    won: bool
    total_steps: int    # env steps only (worker turns), not counting thinker/verifier_role turns

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_goal": self.task_goal,
            "plan": self.plan,
            "turns": [t.to_dict() for t in self.turns],
            "won": self.won,
            "total_steps": self.total_steps,
        }
