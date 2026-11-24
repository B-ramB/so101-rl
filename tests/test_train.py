"""so101_rl.training and so101-rl-train: run config, env and algo kwargs, the run folder, Ctrl+C.

The `slow` tests train for a few thousand steps (CI skips them). conftest.py points SO101_HOME at a temp dir, so
safety.json there holds the package defaults, and every run goes to pytest's tmp_path.
"""

import json
from datetime import datetime

import numpy as np
import pytest
from so101.paths import FOLLOWER_URDF
from so101.safety import SafetyLimits, Workspace
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from so101_rl.cli import train as cli
from so101_rl.envs.reach import ReachEnv
from so101_rl.training import (
    DEFAULT_TD3_NOISE,
    RunConfig,
    default_name,
    env_id,
    limits_to_dict,
    load_config,
    make_algo_kwargs,
    make_env,
    make_env_kwargs,
    short_env_name,
    train,
)

RUN_FILES = (
    "model.zip",
    "best/best_model.zip",
    "config.json",
    "progress.csv",
    "eval/evaluations.npz",
    "monitor/0.monitor.csv",
)


@pytest.fixture(scope="module")
def urdf():
    if not FOLLOWER_URDF.exists():
        pytest.skip(f"no URDF at {FOLLOWER_URDF}; run `uv run so101-fetch-urdfs`")
    return FOLLOWER_URDF


# --- pure logic ---------------------------------------------------------------------------------------------------


def test_env_id_adds_the_module():
    assert env_id("So101Reach-v0") == "so101_rl.envs:So101Reach-v0"
    assert env_id("other.module:Env-v1") == "other.module:Env-v1"


def test_default_name():
    assert short_env_name("So101Reach-v0") == "reach"
    assert short_env_name("so101_rl.envs:So101Lift-v2") == "lift"
    now = datetime(2026, 11, 23, 20, 15, 0)
    assert default_name("So101Reach-v0", "sac", 3, now) == "reach-sac-s3-20261123-201500"
    assert RunConfig(algo="td3", seed=2).name.startswith("reach-td3-s2-")


def test_n_envs_defaults_per_algo():
    assert RunConfig(algo="sac").n_envs == 1
    assert RunConfig(algo="td3").n_envs == 1
    assert RunConfig(algo="ppo").n_envs == 8
    assert RunConfig(algo="ppo", n_envs=2).n_envs == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"algo": "dqn"}, "algo"),
        ({"steps": 0}, "steps"),
        ({"n_envs": 0}, "n_envs"),
        ({"env_kwargs": {"limits": {}}}, "safety.json"),
        ({"env_kwargs": {"render_mode": "human"}}, "render"),
        ({"name": "a/b"}, "folder name"),
        ({"name": ".."}, "folder name"),
    ],
)
def test_run_config_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RunConfig(**kwargs)


def test_env_kwargs_workspace_dicts_become_workspaces():
    kwargs = make_env_kwargs({"workspace": {"z_min": 0.05}, "goal_region": {"r_max": 0.4}, "action_cost": 0.05})
    assert kwargs == {"workspace": Workspace(z_min=0.05), "goal_region": Workspace(r_max=0.4), "action_cost": 0.05}
    with pytest.raises(ValueError, match="unknown keys"):
        make_env_kwargs({"workspace": {"height": 1}})


def test_algo_kwargs_action_noise():
    td3 = make_algo_kwargs("td3", {}, n_actions=5)["action_noise"]
    assert isinstance(td3, NormalActionNoise)
    np.testing.assert_array_equal(td3._sigma, np.full(5, DEFAULT_TD3_NOISE))
    sac = make_algo_kwargs("sac", {"action_noise": 0.2, "learning_rate": 1e-3}, n_actions=5)
    np.testing.assert_array_equal(sac["action_noise"]._sigma, np.full(5, 0.2))
    assert sac["learning_rate"] == 1e-3
    assert "action_noise" not in make_algo_kwargs("sac", {}, n_actions=5)
    with pytest.raises(ValueError, match="PPO"):
        make_algo_kwargs("ppo", {"action_noise": 0.1}, n_actions=5)


def test_limits_round_trip_through_json():
    limits = SafetyLimits.defaults().with_workspace(Workspace(z_min=0.05, r_max=0.4))
    assert SafetyLimits.from_dict(json.loads(json.dumps(limits_to_dict(limits)))) == limits


def test_train_refuses_an_existing_run(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError, match="taken"):
        train(RunConfig(name="taken"), tmp_path)


@pytest.mark.parametrize("flag", ["--env-kwargs", "--algo-kwargs"])
@pytest.mark.parametrize("value", ["{not json", "[1, 2]"])
def test_cli_rejects_bad_json(flag, value, capsys):
    with pytest.raises(SystemExit):
        cli.main([flag, value])
    assert flag in capsys.readouterr().err


def test_cli_reports_an_existing_run(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="taken"):
        cli.main(["--name", "taken", "--runs-dir", str(tmp_path)])


# --- with the sim -------------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_bad_env_kwargs_fail_before_the_run_folder_exists(urdf, tmp_path):
    with pytest.raises(TypeError, match="no_such_parameter"):
        train(RunConfig(name="bad", env_kwargs={"no_such_parameter": 1}), tmp_path)
    assert not (tmp_path / "bad").exists()


@pytest.mark.sim
def test_make_env_uses_the_run_kwargs_and_limits(urdf):
    limits = SafetyLimits.defaults()
    env = make_env("So101Reach-v0", {"action_cost": 0.05, "workspace": {"z_min": 0.05}}, limits)
    try:
        reach = env.unwrapped
        assert isinstance(reach, ReachEnv)
        assert reach.action_cost == 0.05
        assert reach.limits.workspace.z_min == 0.05  # narrowed from safety.json's 0.01
    finally:
        env.close()


class InterruptAt(BaseCallback):
    """Raises KeyboardInterrupt (what Ctrl+C does) after `steps` env steps."""

    def __init__(self, steps: int):
        super().__init__()
        self.steps = steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.steps:
            raise KeyboardInterrupt
        return True


@pytest.mark.slow
@pytest.mark.sim
def test_train_sac_writes_a_complete_run(urdf, tmp_path):
    cfg = RunConfig(
        name="sac", steps=2_000, n_envs=2, eval_every=1_000, eval_episodes=2, algo_kwargs={"learning_starts": 200}
    )
    run = train(cfg, tmp_path)

    assert run == tmp_path / "sac"
    for name in RUN_FILES:
        assert (run / name).is_file(), name
    assert list((run / "tb").glob("events.out.tfevents.*"))
    assert (run / "monitor" / "1.monitor.csv").is_file()  # one per env

    config = load_config(run)
    assert config["env_id"] == "so101_rl.envs:So101Reach-v0"
    assert config["algo_kwargs"] == {"learning_starts": 200}
    assert SafetyLimits.from_dict(config["safety_limits"]) == SafetyLimits.load()
    result = config["result"]
    assert result["steps_done"] == 2_000 and not result["interrupted"]
    assert result["evaluations"] == 2
    assert 0.0 <= result["last_success_rate"] <= 1.0

    env = ReachEnv()
    try:
        obs, _ = env.reset(seed=0)
        for path in (run / "model.zip", run / "best" / "best_model.zip"):
            action, _ = SAC.load(path, device="cpu").predict(obs, deterministic=True)
            assert action.shape == (5,)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.sim
def test_ctrl_c_saves_the_model(urdf, tmp_path):
    cfg = RunConfig(name="stopped", steps=100_000, algo_kwargs={"learning_starts": 100})
    run = train(cfg, tmp_path, callbacks=[InterruptAt(300)])

    result = load_config(run)["result"]
    assert result["interrupted"]
    assert result["steps_done"] == 300
    assert SAC.load(run / "model.zip", device="cpu").num_timesteps == 300


@pytest.mark.slow
@pytest.mark.sim
@pytest.mark.parametrize(
    ("algo", "cls", "algo_kwargs"),
    [
        ("td3", TD3, {"learning_starts": 100}),
        ("ppo", PPO, {"n_steps": 64, "batch_size": 64}),
    ],
)
def test_train_other_algos(urdf, tmp_path, algo, cls, algo_kwargs):
    cfg = RunConfig(
        name=algo, algo=algo, steps=256, n_envs=2, eval_every=10_000, eval_episodes=1, algo_kwargs=algo_kwargs
    )
    run = train(cfg, tmp_path)
    assert load_config(run)["result"]["steps_done"] >= 256
    assert isinstance(cls.load(run / "model.zip", device="cpu"), cls)
