import glob
import os
import shutil

import textworld_express as twx


# textworld-express, like scienceworld, runs a scala/jvm backend over py4j, so a jdk must be
# reachable (the `java` binary on PATH). this locates one across the environments this project
# runs in -- mac (homebrew, keg-only so not auto-on-PATH) and linux GPU studios (conda or apt
# openjdk) -- so a fresh studio only needs `conda install -y openjdk=17` (or apt), not a manual
# JAVA_HOME export every time.
def _ensure_java_on_path() -> None:
    # if JAVA_HOME is already set to a real jdk, trust it and stop.
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isfile(os.path.join(jh, "bin", "java")):
        os.environ["PATH"] = f"{jh}/bin:" + os.environ.get("PATH", "")
        return
    # search known-good jdk homes FIRST, before any bare PATH `java`: macos ships a stub at
    # /usr/bin/java that shutil.which() finds but that is NOT a usable runtime (it just prints
    # "Unable to locate a Java Runtime"), so a which()-based early return silently picks the
    # broken stub over the real homebrew/conda jdk. prefer explicit homes instead.
    candidates = [
        "/opt/homebrew/opt/openjdk@17",                    # mac homebrew
        os.environ.get("CONDA_PREFIX", ""),                # conda openjdk puts java in $CONDA_PREFIX/bin
        "/usr/lib/jvm/java-17-openjdk-amd64",              # debian/ubuntu apt openjdk-17
    ]
    candidates += sorted(glob.glob("/usr/lib/jvm/*"))      # any other apt-installed jvm
    for home in candidates:
        if home and os.path.isfile(os.path.join(home, "bin", "java")):
            os.environ["JAVA_HOME"] = home
            os.environ["PATH"] = f"{home}/bin:" + os.environ.get("PATH", "")
            return
    # nothing usable in the known homes: fall back to a real PATH java only if it actually runs
    # (guards against the macos stub). if even that fails, error loudly with the fix.
    java = shutil.which("java")
    if java:
        try:
            import subprocess
            subprocess.run([java, "-version"], capture_output=True, check=True)
            return  # a working java is already on PATH
        except (subprocess.CalledProcessError, OSError):
            pass
    raise RuntimeError(
        "TextWorldExpress needs a JDK (the `java` binary) but none was found on PATH or in the "
        "usual locations. On a Lightning AI studio run: conda install -y openjdk=17  (or: "
        "sudo apt-get install -y openjdk-17-jdk). On mac: brew install openjdk@17."
    )


_ensure_java_on_path()

STEP_LIMIT = 50  # coin/simonsays/peckingorder are short-horizon, alfworld-comparable

# games chosen because TALES shows a frozen ~3b model solves them 80-100% (a real substrate
# for the reliability layer to improve). deliberately EXCLUDED "sorting" (instant-death on a
# wrong move, the same trap that made scienceworld unusable) and the harder cooking/twc/
# mapreader/arithmetic games (near-0% for a 3b model, no substrate to measure).
#
# re-examined 2026-07-18 with claude-sonnet-5 as worker/thinker instead of a frozen 3b model:
# simonsays is now nearly solved (97% win rate, 50-episode sample) -- little failure signal left
# for a frontier model. HARDER_GAMES below (previously excluded for being near-0% for 3b) were
# tested as a candidate now that the worker model is far stronger. "sorting" stays excluded
# regardless of model strength: instant-death on a wrong move is a property of the environment,
# not a capability gap (same reasoning that disqualified scienceworld's "focus on X").
#
# twc re-tested at step_limit=100 (double default) to rule out a horizon issue: still 0/10 won,
# and trace inspection shows why -- the agent repeats the SAME wrong object placement verbatim
# (e.g. razor -> dressing table, penalized -0.125) for the full 100 steps with zero within-episode
# adaptation to the penalty signal. this is a genuine comprehension/adaptation gap, not a
# step-limit problem, but it's a narrow, repetitive failure signature (one wrong guess repeated)
# rather than diverse failure modes -- dropped from the training mix on that basis.
# verifier-training mix (2026-07-24, difficulty-calibrated). cookingworld dropped: 0% win for a
# 3b worker means all-failure, no success contrast, and it was ~28% of turns dragging the verifier
# toward flat predictions. the remaining games are tuned (see _GAME_PARAMS) so a Qwen2.5-3B worker
# wins ~30-60% -- balanced success/failure, which is what the turn-level verifier needs to learn
# discrimination from and to be EVALUATED on (mapreader, already ~30-40% win, was the one game
# where the trained verifier discriminated well, AUC ~0.78; the others were too easy to produce
# failure examples). peckingorder kept at its default ~90% win despite having no difficulty knob:
# some failure signal plus task variety is better than an empty slot.
TRAINING_GAMES = ("coin", "simonsays", "peckingorder", "mapreader")
DEFAULT_GAMES = ("coin", "simonsays", "peckingorder")
HARDER_GAMES = ("cookingworld", "twc", "mapreader", "arithmetic")
# difficulty params calibrated via scripts/difficulty_sweep.py to land in the ~30-60% win zone:
# coin numLocations=8 (~40%), simonsays gameLength=15 (~60%). mapreader default already balanced;
# peckingorder has no difficulty param.
_GAME_PARAMS = {
    "coin": "numLocations=8,numDistractorItems=5",
    "simonsays": "gameLength=15",
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
