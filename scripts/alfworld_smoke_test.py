"""Confirm ALFWorld installs cleanly and produces a programmatic success/fail signal."""
import os
import yaml

import alfworld.agents.environment as environment

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

with open(os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")) as f:
    config = yaml.safe_load(f)

env_type = config["env"]["type"]
env = environment.get_environment(env_type)(config, train_eval="train")
env = env.init_env(batch_size=1)

obs, info = env.reset()
print("TASK:", obs[0].split("\n")[0])

done = False
steps = 0
while not done and steps < 10:
    admissible = info["admissible_commands"][0]
    action = admissible[0]
    obs, scores, dones, info = env.step([action])
    done = dones[0]
    steps += 1
    print(f"step {steps}: action={action!r} reward={scores[0]} done={done}")

print("EPISODE WON:", info["won"][0])
