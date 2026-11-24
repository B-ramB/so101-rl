"""Train a Stable-Baselines3 policy on an so101-rl env and save everything needed to run it again.

    from so101_rl.training import RunConfig, train

    run = train(RunConfig(name="probe", steps=60_000))   # runs/probe/

The command line is `so101-rl-train` (`so101_rl.cli.train`). A run folder holds:

    runs/<name>/
        model.zip               the policy at the end (also after Ctrl+C)
        best/best_model.zip     the best policy by mean eval reward (EvalCallback, deterministic episodes)
        config.json             env, env kwargs, algo, seed, safety limits, versions, result: see `RunConfig`
        progress.csv            every logged metric (SB3 logger)
        tb/                     TensorBoard events: `uv run tensorboard --logdir runs`
        eval/evaluations.npz    every evaluation (EvalCallback)
        monitor/<rank>.monitor.csv   every training episode: reward, length, is_success, final distance

`train` takes extra SB3 callbacks, extra logger output formats and a callback to run after each evaluation, so other
tools (e.g. a visualizer) can hook into a run without changing this module.
"""

import contextlib
import dataclasses
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from so101.paths import find_project_root
from so101.safety import SafetyLimits, Workspace
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import KVWriter, Logger, make_output_format
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

ALGOS: dict[str, type[BaseAlgorithm]] = {"sac": SAC, "td3": TD3, "ppo": PPO}
# Off-policy algorithms learn from one env well; PPO needs many parallel envs for its rollouts.
DEFAULT_N_ENVS = {"sac": 1, "td3": 1, "ppo": 8}
# SB3's TD3 has no exploration noise by default and then barely explores.
DEFAULT_TD3_NOISE = 0.1
ENV_MODULE = "so101_rl.envs"
EVAL_SEED_OFFSET = 10_000  # the eval env's seed is the run's seed + this, so it never sees the training episodes
CONFIG_FORMAT = 1
# Env kwargs that take a `Workspace`: in JSON they are given as a dict, e.g. {"workspace": {"z_min": 0.05}}.
WORKSPACE_KWARGS = ("workspace", "goal_region")
PACKAGES = ("so101-rl", "so101-core", "stable-baselines3", "gymnasium", "torch", "mujoco")


@dataclasses.dataclass
class RunConfig:
    """What to train. Everything here is written to config.json, so a run can be rebuilt from it.

    env: a registered env id; a bare id (`So101Reach-v0`) is looked up in `so101_rl.envs` (`env_id`).
    env_kwargs, algo_kwargs: JSON values, passed to the env and to the SB3 algorithm. `workspace` / `goal_region` env
        kwargs are dicts of `Workspace` bounds. `action_noise` (SAC, TD3) is the sigma of Gaussian action noise.
    n_envs: parallel envs; None = 1 for SAC and TD3, 8 for PPO. More than 1 runs them in subprocesses.
    eval_every: env steps between evaluations of `eval_episodes` deterministic episodes.
    name: the run folder name; None = `default_name`.
    """

    env: str = "So101Reach-v0"
    algo: str = "sac"
    steps: int = 300_000
    n_envs: int | None = None
    seed: int = 0
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    algo_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    device: str = "auto"
    eval_every: int = 10_000
    eval_episodes: int = 10
    name: str | None = None

    def __post_init__(self):
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {sorted(ALGOS)}, got {self.algo!r}")
        if self.n_envs is None:
            self.n_envs = DEFAULT_N_ENVS[self.algo]
        for field in ("steps", "n_envs", "eval_every", "eval_episodes"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be at least 1, got {getattr(self, field)}")
        if "limits" in self.env_kwargs:
            raise ValueError("env_kwargs cannot set `limits`: the soft limits come from so101_config/safety.json.")
        if "render_mode" in self.env_kwargs:
            raise ValueError("env_kwargs cannot set `render_mode`: training envs never render.")
        if self.name is None:
            self.name = default_name(self.env, self.algo, self.seed)
        if not self.name or self.name in (".", "..") or "/" in self.name or os.sep in self.name:
            raise ValueError(f"name must be a plain folder name, got {self.name!r}")


def env_id(name: str) -> str:
    """The id to make: `So101Reach-v0` -> `so101_rl.envs:So101Reach-v0`. Ids with a module part are kept.

    The `module:id` form makes `gym.make` import the module (which registers the env) first, also in SubprocVecEnv
    workers: their default start method, forkserver, does not inherit the registry.
    """
    return name if ":" in name else f"{ENV_MODULE}:{name}"


def short_env_name(name: str) -> str:
    """`So101Reach-v0` -> `reach`: for run names."""
    bare = name.rsplit(":", 1)[-1]
    return re.sub(r"-v\d+$", "", re.sub(r"^So101", "", bare)).lower() or bare.lower()


def default_name(env: str, algo: str, seed: int, now: datetime | None = None) -> str:
    """`reach-sac-s0-20261123-201500`: env, algo, seed and the start time."""
    now = now or datetime.now()
    return f"{short_env_name(env)}-{algo}-s{seed}-{now:%Y%m%d-%H%M%S}"


def default_runs_dir() -> Path:
    """runs/ at the root of the project you run in (git-ignored)."""
    return find_project_root() / "runs"


def make_env_kwargs(env_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON env kwargs as Python values: `workspace` / `goal_region` dicts become `Workspace`s."""
    kwargs = dict(env_kwargs)
    for key in WORKSPACE_KWARGS:
        if isinstance(kwargs.get(key), Mapping):
            kwargs[key] = Workspace.from_dict(kwargs[key])
    return kwargs


def make_algo_kwargs(algo: str, algo_kwargs: Mapping[str, Any], n_actions: int) -> dict[str, Any]:
    """The JSON algo kwargs as Python values: `action_noise` (a sigma) becomes `NormalActionNoise`. TD3 defaults to
    `DEFAULT_TD3_NOISE`."""
    kwargs = dict(algo_kwargs)
    sigma = kwargs.pop("action_noise", DEFAULT_TD3_NOISE if algo == "td3" else None)
    if sigma is not None:
        if algo == "ppo":
            raise ValueError("action_noise only applies to SAC and TD3; PPO explores through its policy.")
        kwargs["action_noise"] = NormalActionNoise(np.zeros(n_actions), np.full(n_actions, float(sigma)))
    return kwargs


def limits_to_dict(limits: SafetyLimits) -> dict:
    """The limits as the JSON of safety.json; `SafetyLimits.from_dict` reads it back."""
    data = dataclasses.asdict(limits)
    data["joint_limits"] = {joint: list(pair) for joint, pair in data["joint_limits"].items()}
    return data


def make_env(env: str, env_kwargs: Mapping[str, Any], limits: SafetyLimits, **overrides: Any) -> gym.Env:
    """One env as a run trains on it: the run's env id, JSON env kwargs and limits. `overrides` are passed on too,
    e.g. `render_mode="human"` to watch it."""
    return gym.make(env_id(env), **make_env_kwargs(env_kwargs), limits=limits, **overrides)


def make_envs(cfg: RunConfig, limits: SafetyLimits, n_envs: int, seed: int, monitor_dir: Path | None = None) -> VecEnv:
    """`n_envs` copies of the env with `limits`, each wrapped in a Monitor; in subprocesses when there is more than 1.

    `limits` is passed to every env, so all of them use the limits loaded once in this process (and recorded in
    config.json) instead of each reading safety.json. `render_mode` is None: SB3 would otherwise ask for "rgb_array".
    """
    return make_vec_env(
        env_id(cfg.env),
        n_envs=n_envs,
        seed=seed,
        env_kwargs=make_env_kwargs(cfg.env_kwargs) | {"limits": limits, "render_mode": None},
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        monitor_dir=str(monitor_dir) if monitor_dir is not None else None,
        monitor_kwargs={"info_keywords": ("is_success", "distance")},
    )


def train(
    cfg: RunConfig,
    runs_dir: Path | None = None,
    *,
    limits: SafetyLimits | None = None,
    callbacks: Sequence[BaseCallback] = (),
    output_formats: Sequence[KVWriter] = (),
    after_eval: BaseCallback | None = None,
) -> Path:
    """Train `cfg` into `runs_dir/<name>/` and return that folder. Refuses a folder that exists.

    limits: None loads so101_config/safety.json (creating it on first use) here, before any env is made.
    callbacks: extra SB3 callbacks for `learn`. output_formats: extra SB3 logger outputs (next to stdout, CSV and
        TensorBoard). after_eval: a callback run after each evaluation (EvalCallback's `callback_after_eval`).

    Ctrl+C stops training and still saves model.zip; config.json then has `"interrupted": true`.
    """
    run = Path(runs_dir if runs_dir is not None else default_runs_dir()) / cfg.name
    if run.exists():
        raise FileExistsError(f"{run} exists; pick another --name (runs are never overwritten).")
    limits = limits if limits is not None else SafetyLimits.load()

    # The eval env first, in this process: bad env kwargs fail here with a clear error, before any worker starts.
    eval_env = make_envs(cfg, limits, n_envs=1, seed=cfg.seed + EVAL_SEED_OFFSET)
    train_env = None
    created = started = False
    try:
        run.mkdir(parents=True)
        created = True
        train_env = make_envs(cfg, limits, n_envs=cfg.n_envs, seed=cfg.seed, monitor_dir=run / "monitor")
        algo_kwargs = make_algo_kwargs(cfg.algo, cfg.algo_kwargs, train_env.action_space.shape[0])
        model = ALGOS[cfg.algo]("MlpPolicy", train_env, seed=cfg.seed, device=cfg.device, verbose=1, **algo_kwargs)
        model.set_logger(
            Logger(
                str(run),
                [
                    make_output_format("stdout", str(run)),
                    make_output_format("csv", str(run)),
                    make_output_format("tensorboard", str(run / "tb")),
                    *output_formats,
                ],
            )
        )
        evaluation = EvalCallback(
            eval_env,
            n_eval_episodes=cfg.eval_episodes,
            eval_freq=max(cfg.eval_every // cfg.n_envs, 1),  # counted in vec env steps, cfg.n_envs env steps each
            best_model_save_path=str(run / "best"),
            log_path=str(run / "eval"),
            deterministic=True,
            callback_after_eval=after_eval,
            verbose=1,
        )
        config = _config_json(cfg, limits, model)
        _write_json(run / "config.json", config)
        started = True

        start = time.perf_counter()
        interrupted = False
        with warnings.catch_warnings():
            # The eval env runs in this process (DummyVecEnv) while the training envs may be in subprocesses.
            warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")
            try:
                model.learn(cfg.steps, callback=CallbackList([evaluation, *callbacks]))
            except KeyboardInterrupt:
                interrupted = True
                print(f"\nInterrupted after {model.num_timesteps:,} steps; saving the model so far.", flush=True)
        model.save(run / "model.zip")
        config["result"] = _result(model, evaluation, interrupted, time.perf_counter() - start)
        _write_json(run / "config.json", config)
        return run
    except BaseException:
        if created and not started:  # nothing trained yet: don't leave a folder that blocks the same name
            shutil.rmtree(run, ignore_errors=True)
        raise
    finally:
        for env in (train_env, eval_env):
            if env is not None:
                # After Ctrl+C the subprocess workers have exited (they get the SIGINT too), so close can fail.
                with contextlib.suppress(EOFError, BrokenPipeError, ConnectionResetError, OSError):
                    env.close()


def load_config(run: Path) -> dict:
    """A run's config.json."""
    return json.loads((Path(run) / "config.json").read_text())


def _config_json(cfg: RunConfig, limits: SafetyLimits, model: BaseAlgorithm) -> dict:
    return {
        "format": CONFIG_FORMAT,
        "name": cfg.name,
        "env_id": env_id(cfg.env),
        "env_kwargs": cfg.env_kwargs,
        "algo": cfg.algo,
        "algo_kwargs": cfg.algo_kwargs,
        "seed": cfg.seed,
        "n_envs": cfg.n_envs,
        "steps": cfg.steps,
        "eval": {"every": cfg.eval_every, "episodes": cfg.eval_episodes, "seed": cfg.seed + EVAL_SEED_OFFSET},
        "device": str(model.device),
        "safety_limits": limits_to_dict(limits),
        "started": datetime.now().isoformat(timespec="seconds"),
        "versions": _versions(),
        "git": {"so101-rl": _git_commit(Path(__file__)), "so101-core": _git_commit(Path(_core_file()))},
        "result": None,
    }


def _result(model: BaseAlgorithm, evaluation: EvalCallback, interrupted: bool, wall_time_s: float) -> dict:
    successes = [sum(episodes) / len(episodes) for episodes in evaluation.evaluations_successes if episodes]
    return {
        "steps_done": int(model.num_timesteps),
        "interrupted": interrupted,
        "wall_time_s": round(wall_time_s, 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "evaluations": len(evaluation.evaluations_results),
        "best_mean_reward": _finite_or_none(evaluation.best_mean_reward),
        "last_mean_reward": _finite_or_none(evaluation.last_mean_reward),
        "best_success_rate": max(successes) if successes else None,
        "last_success_rate": successes[-1] if successes else None,
    }


def _finite_or_none(value: float) -> float | None:
    """EvalCallback starts its rewards at -inf, which JSON cannot hold."""
    return float(value) if np.isfinite(value) else None


def _write_json(path: Path, data: dict) -> None:
    """Write atomically, so a reader never sees half a file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _core_file() -> str:
    import so101

    return so101.__file__


def _git_commit(path: Path) -> str | None:
    """The commit of the git checkout containing `path`, with `-dirty` for uncommitted changes; None outside git."""
    folder = path.resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit
