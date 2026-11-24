"""so101_rl.viz: Rerun recordings of training runs, checked by reading the .rrd files back (no viewer needed).

The sim tests need the URDFs (skipped otherwise); the slow test trains two tiny runs and compares them.
"""

import collections
import csv
import json
import socket
import time

import numpy as np
import pytest
import rerun as rr
from so101.paths import FOLLOWER_URDF
from so101.safety import SafetyLimits

from so101_rl.cli import compare as compare_cli
from so101_rl.envs.reach import ReachEnv
from so101_rl.training import RunConfig, train
from so101_rl.viz.blueprint import blueprint, loss_keys
from so101_rl.viz.compare import differences, load_runs, read_episodes, read_progress, runs_markdown
from so101_rl.viz.metrics import RerunOutputFormat, metric_path
from so101_rl.viz.recording import (
    RUN_COLORS,
    close,
    connect,
    entity_name,
    new_recording,
    run_color,
    spawn_viewer,
)
from so101_rl.viz.rollout import Player, RolloutRecorder, play
from so101_rl.viz.training import RRD_NAME, RerunTraining, run_markdown


def read(path) -> dict[str, tuple[int, set[str]]]:
    """{entity path: (rows, timelines)} of an .rrd file, without log_time."""
    rows: collections.Counter = collections.Counter()
    timelines: dict[str, set[str]] = collections.defaultdict(set)
    for chunk in rr.chunk.RrdReader(path).stream().to_chunks():
        rows[chunk.entity_path] += chunk.num_rows
        timelines[chunk.entity_path].update(t for t in chunk.timeline_names if t != "log_time")
    return {path: (rows[path], timelines[path]) for path in rows}


def recording(tmp_path, name="run-a"):
    rec = new_recording(name)
    path = tmp_path / f"{name}.rrd"
    connect(rec, path, view=False, blueprint=blueprint(["sac"]))
    return rec, path


@pytest.fixture(scope="module")
def urdf():
    if not FOLLOWER_URDF.exists():
        pytest.skip(f"no URDF at {FOLLOWER_URDF}; run `uv run so101-fetch-urdfs`")
    return FOLLOWER_URDF


# --- pure logic ---------------------------------------------------------------------------------------------------


def test_names_and_paths():
    assert entity_name("reach-sac-s0_20261123") == "reach-sac-s0_20261123"
    assert entity_name("a/b c:d") == "a_b_c_d"
    assert metric_path("eval/success_rate", "run a") == "metrics/eval/success_rate/run_a"


def test_run_colors_are_never_cycled():
    assert run_color(0) == (0x39, 0x87, 0xE5)
    assert len({run_color(i) for i in range(len(RUN_COLORS))}) == len(RUN_COLORS)
    with pytest.raises(ValueError, match="at most"):
        run_color(len(RUN_COLORS))


def test_loss_keys_merge_algorithms():
    keys = loss_keys(["sac", "td3"])
    assert keys.count("train/actor_loss") == 1
    assert "train/ent_coef" in keys and keys[-2:] == ["train/learning_rate", "time/fps"]


def test_output_format_logs_numbers_on_train_step(tmp_path):
    rec, path = recording(tmp_path)
    rec.set_time("rollout_frame", sequence=7)  # left over from a rollout: must not leak onto the curves
    output = RerunOutputFormat(rec, "run-a", run_color(0))
    values = {
        "eval/success_rate": 0.5,
        "rollout/ep_rew_mean": np.float32(-3.5),
        "time/fps": 250,
        "train/note": "text",
        "train/flag": True,
        "train/hidden": 1.0,
        "train/nan": float("nan"),
    }
    output.write(values, {"train/hidden": ("rerun",), "time/fps": ("tensorboard",)}, step=1_000)
    output.write({"eval/success_rate": 0.75}, {}, step=2_000)
    close(rec)

    entities = read(path)
    rows, timelines = entities["/metrics/eval/success_rate/run-a"]
    assert rows == 3  # the style, then two values
    assert timelines == {"train_step"}
    assert "/metrics/rollout/ep_rew_mean/run-a" in entities
    assert "/metrics/time/fps/run-a" in entities  # only excluded from TensorBoard
    for skipped in ("note", "flag", "hidden", "nan"):
        assert f"/metrics/train/{skipped}/run-a" not in entities


def test_read_progress_and_episodes(tmp_path):
    with (tmp_path / "progress.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time/total_timesteps", "eval/success_rate", "rollout/ep_rew_mean"])
        writer.writerows([[1000, "", -20.0], [2000, 0.5, ""], ["", 0.9, -1.0]])
    progress = read_progress(tmp_path)
    assert progress["eval/success_rate"] == ([2000], [0.5])
    assert progress["rollout/ep_rew_mean"] == ([1000], [-20.0])

    monitor = tmp_path / "monitor"
    monitor.mkdir()
    for rank, rows in enumerate([[(1.0, 120, 0.3), (3.0, 120, 0.1)], [(2.0, 120, 0.2)]]):
        with (monitor / f"{rank}.monitor.csv").open("w") as f:
            f.write('#{"t_start": 0, "env_id": "x"}\nr,l,t,is_success,distance\n')
            f.writelines(f"-1.0,{length},{t},False,{d}\n" for t, length, d in rows)
    assert read_episodes(tmp_path) == ([120, 240, 360], [0.3, 0.2, 0.1])  # ordered by end time, over both envs


def test_runs_with_the_same_name_get_their_own_entities(tmp_path):
    for folder in ("x/run", "y/run"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "config.json").write_text(json.dumps({"name": "run", "algo": "sac"}))
    runs = load_runs([tmp_path / "x/run", tmp_path / "y/run"])
    assert [run.name for run in runs] == ["run", "run"]
    assert [run.entity for run in runs] == ["run", "run_"]
    assert runs[0].color != runs[1].color
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_runs([tmp_path])


def test_differences_and_runs_table():
    base = {
        "format": 1,
        "name": "a",
        "started": "x",
        "algo": "sac",
        "seed": 0,
        "env_kwargs": {"action_cost": 0.01},
        "algo_kwargs": {},
        "result": {
            "steps_done": 1000,
            "interrupted": False,
            "wall_time_s": 60.0,
            "last_success_rate": 0.5,
            "best_success_rate": 0.75,
        },
    }
    other = base | {"name": "b", "seed": 1, "env_kwargs": {"action_cost": 0.05}, "algo_kwargs": {"batch_size": 512}}
    assert differences([base, other]) == ["algo_kwargs.batch_size", "env_kwargs.action_cost", "seed"]
    assert differences([base, base | {"name": "c", "started": "y"}]) == []

    class FakeRun:
        def __init__(self, config):
            self.name, self.config, self.finals = config["name"], config, [0.02, 0.04]

    table = runs_markdown([FakeRun(base), FakeRun(other)], race_seeds=(1000, 1001))
    assert "`env_kwargs.action_cost`" in table and "`algo_kwargs.batch_size`" in table
    assert "| a | SAC | 0 | 1,000 | 1.0 min | 50% / 75% | - | 0.01 | 3.0 cm |" in table


# --- with the sim -------------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_rollout_recorder_logs_every_frame(urdf, tmp_path):
    rec, path = recording(tmp_path)
    env = ReachEnv()
    try:
        recorder = RolloutRecorder(rec, series="run-a")
        rng = np.random.default_rng(0)
        player = Player(recorder, env, lambda obs: rng.uniform(-1, 1, 5))
        next_frame, (info,) = play(rec, [player], seed=1000, frame=10, train_step=500)
    finally:
        env.close()
        close(rec)

    frames = env.max_steps + 1  # the reset, then every step
    assert next_frame == 10 + frames
    # Rerun batches the rows of an entity into a few chunks; one chunk per frame (a flush every frame) doubled the files.
    joint_chunks = [c for c in rr.chunk.RrdReader(path).stream().to_chunks() if c.entity_path.endswith("/shoulder_pan")]
    assert len(joint_chunks) < frames // 4
    assert info["distance"] > 0
    entities = read(path)
    rows, timelines = entities["/rollout/joint_transforms/shoulder_pan"]
    assert rows == frames
    assert timelines == {"rollout_frame", "episode_time", "train_step"}
    assert entities["/rollout_plots/distance/run-a"][0] == frames + 1  # + the style
    assert entities["/rollout_plots/action/wrist_roll"][0] == env.max_steps
    for path in ("/rollout/goal", "/rollout/trail", "/rollout/tip", "/rollout_plots/log"):
        assert path in entities
    assert {f"/rollout_plots/reward_terms/{term}" for term in ("distance", "success", "action")} <= entities.keys()


@pytest.mark.sim
def test_players_in_lockstep_get_the_same_goal(urdf, tmp_path):
    rec, _ = recording(tmp_path)
    envs = [ReachEnv(), ReachEnv()]
    try:
        players = [
            Player(
                RolloutRecorder(
                    rec,
                    arm=f"race_{i}",
                    series=f"run{i}",
                    plots="race_plots",
                    offset=(0, i, 0),
                    detail=False,
                    label=f"run{i}",
                ),
                env,
                lambda obs: np.zeros(5),
            )
            for i, env in enumerate(envs)
        ]
        play(rec, players, seed=1001, frame=0)
        np.testing.assert_array_equal(envs[0].goal, envs[1].goal)
    finally:
        for env in envs:
            env.close()
        close(rec)


@pytest.mark.slow
@pytest.mark.sim
def test_train_with_rerun_then_compare(urdf, tmp_path):
    limits = SafetyLimits.load()
    configs = [
        RunConfig(name="a", steps=1_500, eval_every=1_000, eval_episodes=1, algo_kwargs={"learning_starts": 200}),
        RunConfig(
            name="b",
            seed=1,
            steps=1_500,
            eval_every=1_000,
            eval_episodes=1,
            env_kwargs={"action_cost": 0.05},
            algo_kwargs={"learning_starts": 200},
        ),
    ]
    for cfg in configs:
        hooks = RerunTraining(cfg, limits, seeds=(1000,))
        try:
            train(cfg, tmp_path, limits=limits, **hooks.train_kwargs())
        finally:
            hooks.close()

    entities = read(tmp_path / "a" / RRD_NAME)
    assert entities["/metrics/eval/success_rate/a"][1] == {"train_step"}
    assert "/metrics/train/actor_loss/a" in entities
    assert "/episodes/final_distance/a" in entities
    assert entities["/rollouts/final_distance/seed_1000"][0] == 1 + 1  # the style + one evaluation
    assert entities["/rollout/joint_transforms/shoulder_pan"][0] == 121
    assert "/run" in entities

    out = tmp_path / "compare.rrd"
    compare_cli.main([str(tmp_path / "a"), str(tmp_path / "b"), "--episodes", "1", "--save", str(out)])
    entities = read(out)
    for run in ("a", "b"):
        assert f"/metrics/eval/success_rate/{run}" in entities
        assert f"/episodes/final_distance/{run}" in entities
    assert entities["/race_0/joint_transforms/shoulder_pan"][0] == 121
    assert entities["/race_1/joint_transforms/shoulder_pan"][0] == 121
    assert "/run" in entities


def test_run_markdown_shows_the_result():
    config = {
        "name": "a",
        "env_id": "so101_rl.envs:So101Reach-v0",
        "env_kwargs": {},
        "algo": "sac",
        "algo_kwargs": {"learning_rate": 0.001},
        "seed": 0,
        "n_envs": 1,
        "steps": 1000,
        "eval": {"every": 500, "episodes": 2},
        "device": "cpu",
        "started": "2026-11-23T20:00:00",
        "git": {"so101-rl": "abc", "so101-core": "def"},
        "safety_limits": {"joint_limits": {}},
        "result": None,
    }
    page = run_markdown(config)
    assert "# a" in page and '`{"learning_rate": 0.001}`' in page and "Wall time" not in page
    config["result"] = {
        "steps_done": 1000,
        "interrupted": True,
        "wall_time_s": 90.0,
        "last_success_rate": 0.5,
        "best_success_rate": 0.5,
    }
    page = run_markdown(config)
    assert "1,000 (interrupted)" in page and "1.5 min" in page and json.dumps({"joint_limits": {}}, indent=2) in page


def test_spawn_viewer_starts_the_viewer_without_rr_init(monkeypatch):
    """rr.spawn() without rr.init() skips the spawn ("Rerun is disabled"): spawn_viewer must go through the recording."""
    spawned = []
    monkeypatch.setattr(rr.sinks, "_spawn_viewer", lambda **kwargs: spawned.append(kwargs))
    spawn_viewer(new_recording("run-a"))
    assert len(spawned) == 1


def test_close_finishes_the_file_when_the_viewer_is_unreachable(tmp_path, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here: like a viewer that was closed
    path = tmp_path / "run.rrd"
    rec = new_recording("run-a")
    rec.set_sinks(rr.GrpcSink(url=f"rerun+http://127.0.0.1:{port}/proxy"), rr.FileSink(str(path)))
    for step in range(50):
        rec.set_time("train_step", sequence=step)
        rec.log("metrics/x/run-a", rr.Scalars(step))

    start = time.perf_counter()
    close(rec)  # must not raise
    assert time.perf_counter() - start < 30
    assert "viewer not reached" in capsys.readouterr().out
    assert read(path)["/metrics/x/run-a"][0] == 50
