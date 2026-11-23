"""A trained run: its policy, its config, and an env like the one it was trained on.

    model, config = load_run("runs/reach-sac-s0", best=True)
    env = make_run_env(config, render_mode="human")

so101-rl-play (#5) builds on this; it will also compare the run's safety limits with the current safety.json.
"""

from pathlib import Path

import gymnasium as gym
from so101.safety import SafetyLimits
from stable_baselines3.common.base_class import BaseAlgorithm

from so101_rl.training import ALGOS, load_config, make_env


def model_path(run: Path, best: bool = False) -> Path:
    """`runs/<name>/best/best_model.zip` (best) or `runs/<name>/model.zip`."""
    run = Path(run)
    return run / "best" / "best_model.zip" if best else run / "model.zip"


def load_run(run: Path, best: bool = False, device: str = "cpu") -> tuple[BaseAlgorithm, dict]:
    """The SB3 model of a run folder and its config.json. `best` loads the best model by eval reward."""
    run = Path(run)
    config = load_config(run)
    path = model_path(run, best)
    if not path.is_file():
        hint = " (no evaluation finished yet; load the final model instead)" if best else ""
        raise FileNotFoundError(f"No model at {path}{hint}")
    return ALGOS[config["algo"]].load(path, device=device), config


def run_limits(config: dict) -> SafetyLimits:
    """The safety limits the run was trained with (config.json), not the current safety.json."""
    return SafetyLimits.from_dict(config["safety_limits"])


def make_run_env(config: dict, limits: SafetyLimits | None = None, **overrides) -> gym.Env:
    """An env as the run trained on: its env id and env kwargs, with `limits` (default: the run's own)."""
    return make_env(
        config["env_id"], config["env_kwargs"], limits if limits is not None else run_limits(config), **overrides
    )
