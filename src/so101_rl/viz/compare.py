"""Compare training runs in one Rerun recording: curves overlaid, a table of what differs, and a race.

The curves come from each run's progress.csv and monitor/*.csv (written by every run), at the same entity paths as a
live run, so the views are the same and every run is its own line. The race loads each run's policy and plays the
same seeded episodes for all of them in lockstep, arms side by side.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

from so101_rl.policy import load_run, make_run_env, model_path
from so101_rl.training import load_config
from so101_rl.viz.blueprint import blueprint
from so101_rl.viz.metrics import EPISODE_MARKER_SIZE, episode_path, metric_path
from so101_rl.viz.recording import (
    RUN_COLORS,
    TRAIN_STEP,
    close,
    connect,
    entity_name,
    new_recording,
    run_color,
    spawn_viewer,
)
from so101_rl.viz.rollout import Player, RolloutRecorder, play, rollout_seeds, sb3_policy

STEP_COLUMN = "time/total_timesteps"
RACE_SPACING = 0.8  # m between the arms of a race, along y
# config.json keys that are not settings: never listed as a difference.
NOT_SETTINGS = ("format", "name", "started", "result", "versions")


@dataclass
class Run:
    """A run folder to compare, with its display name, entity name and color slot."""

    path: Path
    config: dict
    name: str
    entity: str
    color: tuple[int, int, int]
    finals: list[float] = field(default_factory=list)  # final race distance per seed


def load_runs(paths: Sequence[Path]) -> list[Run]:
    """The runs to compare, each with its own color. Names are the run names, made unique when needed."""
    if not paths:
        raise ValueError("no runs to compare")
    if len(paths) > len(RUN_COLORS):
        raise ValueError(f"compare at most {len(RUN_COLORS)} runs at once (one color each), got {len(paths)}")
    runs: list[Run] = []
    seen: set[str] = set()
    for i, path in enumerate(paths):
        path = Path(path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{path} is not a run folder: no config.json")
        config = load_config(path)
        name = config.get("name") or path.name
        entity = entity_name(name)
        while entity in seen:
            entity += "_"
        seen.add(entity)
        runs.append(Run(path, config, name, entity, run_color(i)))
    return runs


def read_progress(run: Path) -> dict[str, tuple[list[int], list[float]]]:
    """progress.csv as {metric: (train steps, values)}, skipping empty cells and rows without a step."""
    columns: dict[str, tuple[list[int], list[float]]] = {}
    path = Path(run) / "progress.csv"
    if not path.is_file():
        return columns
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = _number(row.get(STEP_COLUMN))
            if step is None:
                continue
            for key, text in row.items():
                value = _number(text)
                if key == STEP_COLUMN or value is None:
                    continue
                steps, values = columns.setdefault(key, ([], []))
                steps.append(int(step))
                values.append(value)
    return columns


def read_episodes(run: Path) -> tuple[list[int], list[float]]:
    """(train step, final distance) of every training episode, from monitor/*.monitor.csv.

    The monitor files have no step column: the step of an episode is the summed length of all episodes (of every env)
    that ended up to it, in order of their end time.
    """
    episodes: list[tuple[float, int, float]] = []  # (end time, length, final distance)
    for path in sorted((Path(run) / "monitor").glob("*.monitor.csv")):
        with path.open(newline="") as f:
            next(f, None)  # the first line is a JSON comment: #{"t_start": ..., "env_id": ...}
            for row in csv.DictReader(f):
                end, length, distance = _number(row.get("t")), _number(row.get("l")), _number(row.get("distance"))
                if end is not None and length is not None and distance is not None:
                    episodes.append((end, int(length), distance))
    episodes.sort()
    steps = np.cumsum([length for _, length, _ in episodes]).tolist()
    return steps, [distance for *_, distance in episodes]


def _number(text: Any) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def log_curves(rec: rr.RecordingStream, run: Run) -> None:
    """A run's metrics and training episodes, at the paths a live run uses."""
    for key, (steps, values) in read_progress(run.path).items():
        path = metric_path(key, run.entity)  # the entity, not the name: two runs may share a name
        rec.log(path, rr.SeriesLines(names=run.name, colors=[run.color], widths=[2.0]), static=True)
        rec.send_columns(
            path, indexes=[rr.TimeColumn(TRAIN_STEP, sequence=steps)], columns=rr.Scalars.columns(scalars=values)
        )
    steps, distances = read_episodes(run.path)
    if steps:
        path = episode_path(run.entity)
        style = rr.SeriesPoints(names=run.name, colors=[run.color], marker_sizes=[EPISODE_MARKER_SIZE])
        rec.log(path, style, static=True)
        rec.send_columns(
            path, indexes=[rr.TimeColumn(TRAIN_STEP, sequence=steps)], columns=rr.Scalars.columns(scalars=distances)
        )


def _flatten(data: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if not prefix and key in NOT_SETTINGS:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict) and name != "safety_limits":  # an empty dict adds no keys
            flat |= _flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


def differences(configs: Sequence[dict]) -> list[str]:
    """The config keys (dotted, e.g. `env_kwargs.action_cost`) whose values are not the same in every run."""
    flats = [_flatten(config) for config in configs]
    keys = sorted({key for flat in flats for key in flat})
    missing = object()
    return [key for key in keys if len({json.dumps(flat.get(key, missing), default=str) for flat in flats}) > 1]


def runs_markdown(runs: Sequence[Run], race_seeds: Sequence[int] = ()) -> str:
    """A table with one row per run: the main facts, then every setting that differs between the runs."""
    shown = ("algo", "seed", "eval.seed")  # algo and seed have their own columns; eval.seed follows seed
    diff = [key for key in differences([run.config for run in runs]) if key not in shown]
    head = ["Run", "Algo", "Seed", "Steps done", "Wall time", "Eval success last / best", *[f"`{k}`" for k in diff]]
    if race_seeds:
        head.append("Race: mean final distance")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in runs:
        flat = _flatten(run.config)
        result = run.config.get("result") or {}
        cells = [
            run.name,
            run.config["algo"].upper(),
            str(run.config["seed"]),
            f"{result['steps_done']:,}" + (" (interrupted)" if result.get("interrupted") else "") if result else "-",
            f"{result['wall_time_s'] / 60:.1f} min" if result else "-",
            _rates(result),
            *[_cell(flat.get(key)) for key in diff],
        ]
        if race_seeds:
            cells.append(f"{np.mean(run.finals) * 100:.1f} cm" if run.finals else "-")
        lines.append("| " + " | ".join(cells) + " |")
    note = ""
    if not diff:
        note = "\n\nNo settings differ apart from algorithm and seed."
    seeds = f"\n\nRace seeds: {', '.join(map(str, race_seeds))}." if race_seeds else ""
    return "# Runs\n\n" + "\n".join(lines) + note + seeds + "\n"


def _rates(result: dict) -> str:
    last, best = result.get("last_success_rate"), result.get("best_success_rate")
    return "-" if last is None else f"{last:.0%} / {best:.0%}"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def race(rec: rr.RecordingStream, runs: Sequence[Run], seeds: Sequence[int], final: bool = False) -> None:
    """Every run's policy on the same seeded episodes, in lockstep, arms side by side along y.

    Uses each run's best model unless `final` (or when a run has none yet), and the env kwargs and safety limits it
    was trained with. Fills `run.finals`.
    """
    players = []
    try:
        for i, run in enumerate(runs):
            best = not final and model_path(run.path, best=True).is_file()
            if not final and not best:
                print(f"{run.name}: no best model yet, using the final one.")
            model, _ = load_run(run.path, best=best)
            env = make_run_env(run.config)
            recorder = RolloutRecorder(
                rec,
                arm=f"race_{i}",
                series=run.entity,
                plots="race_plots",
                offset=(0.0, i * RACE_SPACING, 0.0),
                color=run.color,
                detail=False,
                label=run.name,
            )
            players.append(Player(recorder, env, sb3_policy(model)))
        frame = 0
        for seed in seeds:
            frame, infos = play(rec, players, seed, frame)
            for run, info in zip(runs, infos, strict=True):
                run.finals.append(float(info["distance"]))
    finally:
        for player in players:
            player.env.close()


def compare(
    paths: Sequence[Path], rrd: Path | None = None, view: bool = True, episodes: int = 3, final: bool = False
) -> list[Run]:
    """Write the comparison of `paths` to `rrd` and/or the viewer. `episodes` race episodes (0: no race)."""
    runs = load_runs(paths)
    seeds = rollout_seeds(episodes)
    arms = [f"race_{i}" for i in range(len(runs))] if seeds else []
    rec = new_recording("compare: " + ", ".join(run.name for run in runs))
    if view:
        spawn_viewer()
    connect(rec, rrd, view, blueprint({run.config["algo"] for run in runs}, rollouts=False, race_arms=arms))
    try:
        for run in runs:
            log_curves(rec, run)
        if seeds:
            race(rec, runs, seeds, final)
        rec.log("run", rr.TextDocument(runs_markdown(runs, seeds), media_type=rr.MediaType.MARKDOWN), static=True)
    finally:
        close(rec)
    return runs
