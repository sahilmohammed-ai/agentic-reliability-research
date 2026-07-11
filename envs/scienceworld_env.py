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

# scienceworld returns ~400-500 valid commands per step, dominated by "connect X to Y"
# pairings between every object and every other object (mostly physically meaningless, e.g.
# "connect chair to cloth"), but even the non-connect commands can number ~200 (move/use/pour
# over many objects). a frozen 3b worker drowns in that: it grabs from the top of the sorted
# list and never reasons about the task, giving 0% win rate (build_9, unfiltered). so cap the
# list PER VERB: keep at most PER_VERB_CAP commands of each action type. this guarantees every
# action type stays represented (no task becomes categorically unwinnable, including circuit
# tasks that need "connect"), while bounding total length to something a small model can scan.
# for "connect" specifically, electrical-looking commands are kept first, since those are the
# only connects that win the ~3-4 circuit task types.
PER_VERB_CAP = 8   # max commands of any single verb shown to the worker
_ELECTRICAL_HINTS = (
    "wire", "battery", "bulb", "terminal", "cable", "switch", "anode", "cathode",
    "generator", "motor", "solar", "wind", "led", "light", "power", "electric",
)


def _looks_electrical(command: str) -> bool:
    """true if a connect command references a plausibly-electrical component, so the
    connect commands that can actually win circuit tasks are preferred when capping."""
    return any(h in command.lower() for h in _ELECTRICAL_HINTS)


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
        # advertised to the runner so it caps episodes at scienceworld's horizon, not alfworld's
        self.step_limit = step_limit

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
        prefer info['valid'] (already computed by the last step); fall back to a live query.
        the raw list is then length-capped, see _cap_commands for why."""
        if info and info.get("valid"):
            raw = info["valid"]
        else:
            raw = self._env.get_valid_action_object_combinations()
        return self._cap_commands(raw)

    def _cap_commands(self, raw: list[str]) -> list[str]:
        """bound the command list per verb so no single action type swamps the worker's
        prompt, while keeping every action type represented. keeps at most PER_VERB_CAP of
        each verb; for 'connect' the electrical-looking commands are taken first so the
        circuit-task-winning connects survive. original relative order is preserved."""
        by_verb: dict[str, list[str]] = {}
        for c in raw:
            by_verb.setdefault(c.split()[0], []).append(c)

        # for connect, sort electrical-looking commands to the front before capping
        if "connect" in by_verb:
            conn = by_verb["connect"]
            by_verb["connect"] = (
                [c for c in conn if _looks_electrical(c)]
                + [c for c in conn if not _looks_electrical(c)]
            )

        kept = set()
        for verb, cmds in by_verb.items():
            kept.update(cmds[:PER_VERB_CAP])

        # preserve the original ordering of whatever survived
        return [c for c in raw if c in kept]

    def won(self, info: dict) -> bool:
        """success = full task score. scienceworld's own 'done' also flips on timeout/failure,
        so gate on score == 100 rather than the completion flag to mean a real win."""
        return info.get("score", 0) >= 100

    def task_goal(self, obs: str, info: dict) -> str:
        """the task description, parallel to alfworld's 'Your task is to: ...' line.
        signature is (obs, info) to match AlfWorldEnv; scienceworld puts the goal in info."""
        return info.get("taskDesc", info.get("taskName", ""))

    def worker_hint(self) -> str:
        """scienceworld-specific caveat for the worker. 'focus on X' is an irreversible
        commitment that ends the episode: focusing on the right target scores, focusing on
        the wrong one (or on yourself) ends the episode with a large penalty. a naive worker
        fires 'focus on agent' immediately and dies, so warn it explicitly."""
        return (
            "IMPORTANT: 'focus on X' is a final, irreversible commitment that ends the task. "
            "Only use 'focus on X' when X is exactly the target object the task asks you to focus "
            "on, and only after you have finished the required experiment/steps. Never 'focus on "
            "agent' or 'focus on air'. Focusing on the wrong thing ends the task in failure."
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
