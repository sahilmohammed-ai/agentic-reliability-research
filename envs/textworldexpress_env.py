import os

import textworld_express as twx


# textworld-express, like scienceworld, runs a scala/jvm backend over py4j, so a jdk must be
# reachable. openjdk@17 (homebrew) is keg-only, so point at it explicitly unless the caller
# already set JAVA_HOME. mirrors the scienceworld wrapper's handling.
_DEFAULT_JAVA_HOME = "/opt/homebrew/opt/openjdk@17"
if "JAVA_HOME" not in os.environ and os.path.isdir(_DEFAULT_JAVA_HOME):
    os.environ["JAVA_HOME"] = _DEFAULT_JAVA_HOME
    os.environ["PATH"] = f"{_DEFAULT_JAVA_HOME}/bin:" + os.environ.get("PATH", "")

STEP_LIMIT = 50  # coin/simonsays/peckingorder are short-horizon, alfworld-comparable

# games chosen because TALES shows a frozen ~3b model solves them 80-100% (a real substrate
# for the reliability layer to improve). deliberately EXCLUDES "sorting" (instant-death on a
# wrong move, the same trap that made scienceworld unusable) and the harder cooking/twc/
# mapreader/arithmetic games (near-0% for a 3b model, no substrate to measure).
DEFAULT_GAMES = ("coin", "simonsays", "peckingorder")
# modest game params keep the tasks in the tractable range for a small worker
_GAME_PARAMS = {
    "coin": "numLocations=5,numDistractorItems=3",
    "simonsays": "gameLength=5",
    "peckingorder": "",
}


class TextWorldExpressEnvWrapper:
    """thin wrapper around TextWorldExpress with the same api as AlfWorldEnv, so the runner,
    agents, and masking logic work unchanged. each episode samples a random game from
    DEFAULT_GAMES and a random seed, parallel to how AlfWorldEnv hands out a different game
    each reset."""

    def __init__(self, split: str = "train", games: tuple[str, ...] = DEFAULT_GAMES, step_limit: int = STEP_LIMIT):
        self._env = twx.TextWorldExpressEnv(envStepLimit=step_limit)
        self._games = games
        self._split = split
        self.step_limit = step_limit  # advertised to the runner as the episode cap
        self._rng_seed = 0  # incremented per reset for variation

    def _fold(self) -> str:
        if self._split == "train":
            return "train"
        elif self._split in ("eval_id", "dev"):
            return "dev"
        return "test"

    def reset(self) -> tuple[str, dict]:
        """start a new episode on a randomly chosen game + seed."""
        import random
        game = random.choice(self._games)
        self._env.load(gameName=game, gameParams=_GAME_PARAMS.get(game, ""))
        self._rng_seed += 1
        obs, info = self._env.reset(seed=self._rng_seed, gameFold=self._fold())
        info = dict(info)
        info["gameName"] = game
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """execute one action and return observation, reward, done flag, and info."""
        obs, reward, done, info = self._env.step(action)
        return obs, float(reward), bool(done), dict(info)

    def admissible_commands(self, info: dict) -> list[str]:
        """valid action strings at the current state, analogous to alfworld's list.
        twx returns a modest ~8-command list, no capping needed (unlike scienceworld)."""
        return info.get("validActions", [])

    def won(self, info: dict) -> bool:
        """twx exposes a clean boolean task-success flag, no score-thresholding needed."""
        return bool(info.get("tasksuccess", False))

    def task_goal(self, obs: str, info: dict) -> str:
        """the task description. signature is (obs, info) to match AlfWorldEnv; twx puts the
        goal in info['taskDescription']."""
        return info.get("taskDescription", info.get("gameName", ""))

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
