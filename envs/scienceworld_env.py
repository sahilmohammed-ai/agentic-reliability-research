import os
import random

from scienceworld import ScienceWorldEnv


# scienceworld ships a scala/jvm backend served over py4j, so a jdk must be reachable.
# openjdk@17 from homebrew is keg-only (not on the global PATH), so point at it explicitly
# unless the caller already set JAVA_HOME. this keeps the env self-contained the same way
# alfworld_env defaults ALFWORLD_DATA.
_DEFAULT_JAVA_HOME = "/opt/homebrew/opt/openjdk@17"
if "JAVA_HOME" not in os.environ and os.path.isdir(_DEFAULT_JAVA_HOME):
    os.environ["JAVA_HOME"] = _DEFAULT_JAVA_HOME
    os.environ["PATH"] = f"{_DEFAULT_JAVA_HOME}/bin:" + os.environ.get("PATH", "")

STEP_LIMIT = 100  # scienceworld's own cap; tasks are longer-horizon than alfworld


class ScienceWorldEnvWrapper:
    """thin wrapper around ScienceWorld with the same api as AlfWorldEnv, so the runner,
    agents, and masking logic work against either environment unchanged.

    per episode it samples a random task and a random train-split variation, parallel to
    how AlfWorldEnv hands out a different game each reset."""

    def __init__(self, split: str = "train", step_limit: int = STEP_LIMIT):
        self._env = ScienceWorldEnv("", serverPath="", envStepLimit=step_limit)
        self._task_names = self._env.get_task_names()
        self._split = split
        self._last_info = None

    def _variation_for_split(self, task_name: str) -> int:
        """pick a random variation index from the requested split for this task."""
        self._env.load(task_name, variationIdx=0, simplificationStr="", generateGoldPath=False)
        if self._split == "train":
            return self._env.get_random_variation_train()
        elif self._split in ("eval_id", "dev"):
            return self._env.get_random_variation_dev()
        else:  # eval_ood / test
            return self._env.get_random_variation_test()

    def reset(self) -> tuple[str, dict]:
        """start a new episode on a randomly chosen task + variation."""
        task_name = random.choice(self._task_names)
        variation = self._variation_for_split(task_name)
        self._env.load(task_name, variationIdx=variation, simplificationStr="", generateGoldPath=False)
        obs, info = self._env.reset()
        # surface the task description as the "goal" line callers expect; scienceworld puts
        # it in info['taskDesc'] rather than embedding it in the observation like alfworld
        info = dict(info)
        info["taskName"] = task_name
        self._last_info = info
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """execute one action and return observation, reward, done flag, and info."""
        obs, reward, done, info = self._env.step(action)
        info = dict(info)
        self._last_info = info
        return obs, float(reward), bool(done), info

    def admissible_commands(self, info: dict) -> list[str]:
        """valid action strings at the current state, analogous to alfworld's list.
        prefer info['valid'] (already computed by the last step); fall back to a live query."""
        if info and info.get("valid"):
            return info["valid"]
        return self._env.get_valid_action_object_combinations()

    def won(self, info: dict) -> bool:
        """success = full task score. scienceworld's own 'done' also flips on timeout/failure,
        so gate on score == 100 rather than the completion flag to mean a real win."""
        return info.get("score", 0) >= 100

    def task_goal(self, obs: str, info: dict) -> str:
        """the task description, parallel to alfworld's 'Your task is to: ...' line.
        signature is (obs, info) to match AlfWorldEnv; scienceworld puts the goal in info."""
        return info.get("taskDesc", info.get("taskName", ""))

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
