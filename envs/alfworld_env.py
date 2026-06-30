import os
import yaml
import alfworld.agents.environment as environment


class AlfWorldEnv:
    """Thin wrapper around ALFWorld's TextWorld env with a consistent gym-style API."""

    def __init__(self, config_path: str, split: str = "train"):
        os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        env_type = config["env"]["type"]
        self._env = environment.get_environment(env_type)(config, train_eval=split)
        self._env = self._env.init_env(batch_size=1)
        self._last_info = None

    def reset(self) -> tuple[str, dict]:
        obs, info = self._env.reset()
        self._last_info = info
        return obs[0], info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, scores, dones, info = self._env.step([action])
        self._last_info = info
        return obs[0], float(scores[0]), bool(dones[0]), info

    def admissible_commands(self, info: dict) -> list[str]:
        return info["admissible_commands"][0]

    def won(self, info: dict) -> bool:
        return bool(info["won"][0])

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
