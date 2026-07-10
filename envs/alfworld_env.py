import os
import yaml
import alfworld.agents.environment as environment


# wrapper around ALFWorld that provides a gym-style interface
class AlfWorldEnv:
    """thin wrapper around ALFWorld's TextWorld env with a consistent gym-style API."""

    def __init__(self, config_path: str, split: str = "train"):
        # load yaml config and initialize alfworld environment
        os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        env_type = config["env"]["type"]
        self._env = environment.get_environment(env_type)(config, train_eval=split)
        self._env = self._env.init_env(batch_size=1)
        self._last_info = None

    def reset(self) -> tuple[str, dict]:
        """start new episode and return initial observation."""
        obs, info = self._env.reset()
        self._last_info = info
        return obs[0], info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """execute one action and return observation, reward, done flag, and info."""
        # unwrap batch results since we use batch_size=1
        obs, scores, dones, info = self._env.step([action])
        self._last_info = info
        return obs[0], float(scores[0]), bool(dones[0]), info

    def admissible_commands(self, info: dict) -> list[str]:
        """extract available actions from info dict."""
        return info["admissible_commands"][0]

    def won(self, info: dict) -> bool:
        """check if task was completed successfully."""
        return bool(info["won"][0])

    def task_goal(self, obs: str, info: dict) -> str:
        """the 'Your task is to: ...' line, which alfworld embeds in the observation text.
        signature is (obs, info) to match the scienceworld wrapper (which reads info instead),
        so the runner can call env.task_goal(obs, info) uniformly for either environment."""
        obs_lines = [l.strip() for l in obs.split("\n")]
        return next((l for l in obs_lines if l.startswith("Your task is to:")), obs_lines[0])

    def close(self):
        """clean up environment resources gracefully."""
        try:
            self._env.close()
        except Exception:
            pass
